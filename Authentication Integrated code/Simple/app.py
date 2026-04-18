# This code generate Random OTP on same screen to verify SignUp credential and used for log in

import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from pyvis.network import Network
import tempfile
import streamlit.components.v1 as components

# ---------------- LOAD DATA ----------------
ratings = pd.read_csv("ratings.csv")
movies = pd.read_csv("movies.csv")

# ---------------- BUILD GRAPH ----------------
def build_graph(ratings):
    user_ids = ratings['userId'].unique()
    movie_ids = ratings['movieId'].unique()

    user_map = {int(u): i for i, u in enumerate(user_ids)}
    movie_map = {int(m): i + len(user_ids) for i, m in enumerate(movie_ids)}

    edges, weights = [], []

    for _, row in ratings.iterrows():
        u = user_map[int(row['userId'])]
        m = movie_map[int(row['movieId'])]
        r = row['rating']

        edges.append([u, m])
        edges.append([m, u])
        weights.append(r)
        weights.append(r)

    return (
        torch.tensor(edges).t().contiguous(),
        torch.tensor(weights, dtype=torch.float),
        user_map,
        movie_map,
        len(user_ids) + len(movie_ids)
    )

edge_index, edge_weight, user_map, movie_map, num_nodes = build_graph(ratings)

# ---------------- MODEL ----------------
class GNNRecommender(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.embedding = nn.Embedding(num_nodes, 64)
        self.conv1 = GCNConv(64, 64)
        self.conv2 = GCNConv(64, 32)

    def forward(self, edge_index, edge_weight):
        x = self.embedding.weight
        x = self.conv1(x, edge_index, edge_weight)
        x = torch.relu(x)
        x = self.conv2(x, edge_index, edge_weight)

        # 🔥 NORMALIZATION (IMPORTANT)
        x = F.normalize(x, p=2, dim=1)
        return x

model = GNNRecommender(num_nodes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# ---------------- TRAIN ----------------
def train():
    model.train()
    optimizer.zero_grad()

    emb = model(edge_index, edge_weight)
    src, dst = edge_index

    pos = (emb[src] * emb[dst]).sum(dim=1)
    neg_dst = torch.randint(0, num_nodes, dst.size())
    neg = (emb[src] * emb[neg_dst]).sum(dim=1)

    loss = -torch.log(torch.sigmoid(pos)).mean() - torch.log(1 - torch.sigmoid(neg)).mean()

    loss.backward()
    optimizer.step()

# 🔥 TRAIN LONGER (IMPORTANT)
for _ in range(100):
    train()

# ---------------- SIMILAR USERS ----------------
def get_similar_users(user_id, emb, user_map):
    user_id = int(user_id)
    user_idx = user_map[user_id]
    user_emb = emb[user_idx]

    sims = []

    for u, idx in user_map.items():
        if u != user_id:
            sim = F.cosine_similarity(
                user_emb.unsqueeze(0),
                emb[idx].unsqueeze(0)
            ).item()

            sims.append((int(u), sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)

    return [u for u, _ in sims[:5]]

# ---------------- RECOMMEND ----------------
def recommend(user_id):
    user_id = int(user_id)

    model.eval()
    with torch.no_grad():
        emb = model(edge_index, edge_weight)

    sim_users = get_similar_users(user_id, emb, user_map)

    sim_data = ratings[ratings['userId'].isin(sim_users)]
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].values

    # 🔥 REMOVE WATCHED MOVIES
    rec = sim_data[~sim_data['movieId'].isin(user_movies)]

    # 🔥 AGGREGATE (IMPORTANT FIX)
    rec_grouped = rec.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()

    rec_grouped.columns = ['movieId', 'avg_rating', 'count']

    # 🔥 SCORE = rating + popularity
    rec_grouped['score'] = rec_grouped['avg_rating'] * rec_grouped['count']

    # 🔥 SORT
    rec_grouped = rec_grouped.sort_values(by='score', ascending=False)

    # 🔥 GET MORE MOVIES (10 instead of 5)
    top_movies = rec_grouped['movieId'].head(10)

    return movies[movies['movieId'].isin(top_movies)][['title']], sim_users

# ---------------- RECOMMEND ----------------
def personalized_recommend(user_id, model, edge_index, edge_weight, user_map):
    model.eval()
    with torch.no_grad():
        emb = model(edge_index, edge_weight)

    sim_users = get_similar_users(user_id, emb, user_map)

    sim_data = ratings[ratings['userId'].isin(sim_users)]
    user_movies = ratings[ratings['userId'] == user_id]['movieId'].values

    rec = sim_data[~sim_data['movieId'].isin(user_movies)]

    # 🔥 AGGREGATE
    rec_grouped = rec.groupby('movieId').agg({
        'rating': ['mean', 'count']
    }).reset_index()

    rec_grouped.columns = ['movieId', 'avg_rating', 'count']

    rec_grouped['score'] = rec_grouped['avg_rating'] * rec_grouped['count']

    rec_grouped = rec_grouped.sort_values(by='score', ascending=False)

    top_movies = rec_grouped['movieId'].head(10)

    return movies[movies['movieId'].isin(top_movies)][['title']], sim_users

# ---------------- ADD NEW USER ----------------
def add_user(selected_movies, ratings_input):
    global ratings

    new_user_id = ratings['userId'].max() + 1

    new_data = []
    for m, r in zip(selected_movies, ratings_input):
        new_data.append({
            "userId": new_user_id,
            "movieId": m,
            "rating": r
        })

    ratings = pd.concat([ratings, pd.DataFrame(new_data)], ignore_index=True)
    ratings.to_csv("ratings.csv", index=False)

    return new_user_id

# ---------------- GRAPH ----------------
def show_graph(user_id=None, sim_users=None):
    net = Network(height="650px", width="100%", bgcolor="#0e1117", font_color="white")
    net.barnes_hut()

    if user_id:
        user_id = int(user_id)
    if sim_users:
        sim_users = [int(u) for u in sim_users]

    if user_id:
        focus = ratings[ratings['userId'] == user_id]
        sample = pd.concat([ratings.sample(200), focus])
    else:
        sample = ratings.sample(300)

    for _, row in sample.iterrows():
        uid = int(row['userId'])
        mid = int(row['movieId'])

        user_node = f"User {uid}"
        movie_node = f"Movie {mid}"

        # Color logic
        if uid == user_id:
            color = "yellow"
            size = 25
        elif sim_users and uid in sim_users:
            color = "green"
            size = 18
        else:
            color = "red"
            size = 10

        net.add_node(user_node, color=color, size=size)
        net.add_node(movie_node, color="blue", size=8)

        if row['rating'] >= 3:
            net.add_edge(user_node, movie_node, title=f"Rating: {row['rating']}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)

    html = open(tmp.name).read()
    components.html(html, height=650)

# ---------------- FOCUS GRAPH ----------------
def show_focus(user_id):
    user_id = int(user_id)

    net = Network(height="650px", width="100%", bgcolor="black", font_color="white")

    user_data = ratings[ratings['userId'] == user_id]

    net.add_node(f"User {user_id}", color="yellow", size=30)

    for _, row in user_data.iterrows():
        movie = f"Movie {int(row['movieId'])}"
        net.add_node(movie, color="blue")
        net.add_edge(f"User {user_id}", movie, title=f"Rating: {row['rating']}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)

    html = open(tmp.name).read()
    components.html(html, height=650)
    
# ---------------- FOCUSED GRAPH ----------------
def show_focus_graph(user_id):
    user_id = int(user_id)

    net = Network(height="650px", width="100%", bgcolor="#000", font_color="white")

    net.barnes_hut()

    user_data = ratings[ratings['userId'] == user_id]

    # Add main user
    net.add_node(f"User {user_id}", color="yellow", size=30)

    for _, row in user_data.iterrows():
        mid = int(row['movieId'])
        movie_node = f"Movie {mid}"

        net.add_node(movie_node, color="blue", size=15)

        net.add_edge(
            f"User {user_id}",
            movie_node,
            title=f"Rating: {row['rating']}"
        )

    net.set_options("""
    var options = {
      "interaction": {
        "hover": true
      }
    }
    """)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)

    html = open(tmp.name, "r", encoding="utf-8").read()
    components.html(html, height=650)

# ---------------- USER SIM GRAPH ----------------
def show_user_sim_graph(user_id):
    net = Network(height="650px", width="100%", bgcolor="black", font_color="white")

    model.eval()
    with torch.no_grad():
        emb = model(edge_index, edge_weight)

    user_id = int(user_id)
    user_idx = user_map[user_id]
    target = emb[user_idx]

    sims = []
    for u, idx in user_map.items():
        if u != user_id:
            sim = F.cosine_similarity(
                target.unsqueeze(0),
                emb[idx].unsqueeze(0)
            ).item()
            sims.append((int(u), sim))

    sims = sorted(sims, key=lambda x: x[1], reverse=True)[:5]

    net.add_node(f"User {user_id}", color="yellow", size=30)

    for u, sim in sims:
        net.add_node(f"User {u}", color="green")
        net.add_edge(f"User {user_id}", f"User {u}", title=f"{sim:.3f}")

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
    net.save_graph(tmp.name)

    html = open(tmp.name).read()
    components.html(html, height=650)
    
# ---------------- AUTH SYSTEM ----------------
import random

USER_FILE = "users.csv"

def load_users():
    try:
        return pd.read_csv(USER_FILE)
    except:
        return pd.DataFrame(columns=["email", "password"])

def save_user(email, password):
    df = load_users()
    df = pd.concat([df, pd.DataFrame([{"email": email, "password": password}])])
    df.to_csv(USER_FILE, index=False)

def authenticate(email, password):
    df = load_users()
    user = df[(df['email'] == email) & (df['password'] == password)]
    return not user.empty

def user_exists(email):
    df = load_users()
    return email in df['email'].values

def generate_otp():
    return str(random.randint(1000, 9999))

# ---------------- UI ----------------
# ---------------- UI ----------------
st.title("🎬 GNN Movie Recommender")

if "auth" not in st.session_state:
    st.session_state.auth = None

menu = st.sidebar.selectbox("Select Role", ["Guest", "User Login", "User Signup", "Admin Login"])

# ---------------- GUEST ----------------
if menu == "Guest":
    user_id = st.number_input("Enter User ID", min_value=1, step=1)

    if "recs" not in st.session_state:
        st.session_state.recs = None
    if "sim" not in st.session_state:
        st.session_state.sim = None

    if st.button("Get Recommendations"):
        recs, sim = recommend(user_id)
        st.session_state.recs = recs
        st.session_state.sim = sim

    if st.session_state.recs is not None:
        st.write("### Similar Users")
        st.write(st.session_state.sim)

        st.write("### Recommendations")
        st.table(st.session_state.recs)

        if st.button("🔍 Show Highlight Graph", key="guest_highlight"):
            show_graph(user_id, st.session_state.sim)

        if st.button("🎯 Show Focus Graph", key="guest_focus"):
            show_focus(user_id)

        if st.button("👥 Show User Similarity Graph", key="guest_sim"):
            show_user_sim_graph(user_id)

        if st.button("📊 Show Full Graph", key="guest_full"):
            show_graph()

# ---------------- USER SIGNUP ----------------
elif menu == "User Signup":
    st.header("📝 Signup")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Register"):
        if user_exists(email):
            st.error("User already exists")
        else:
            otp = generate_otp()
            st.session_state.otp = otp
            st.session_state.temp_user = (email, password)

            st.info(f"OTP (for demo): {otp}")

    otp_input = st.text_input("Enter OTP")

    if st.button("Verify OTP"):
        if otp_input == st.session_state.get("otp"):
            email, password = st.session_state.temp_user
            save_user(email, password)
            st.success("Signup successful")
        else:
            st.error("Invalid OTP")

# ---------------- USER LOGIN ----------------
elif menu == "User Login":
    st.header("🔐 Login")

    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if authenticate(email, password):
            st.session_state.auth = email
            st.success("Login successful")
        else:
            st.error("Invalid credentials")

    if st.button("Forgot Password"):
        if user_exists(email):
            new_pass = generate_otp()
            df = load_users()
            df.loc[df['email'] == email, 'password'] = new_pass
            df.to_csv(USER_FILE, index=False)
            st.info(f"New Password (demo): {new_pass}")
        else:
            st.error("Email not found")

    # AFTER LOGIN SHOW USER PAGE
    if st.session_state.auth:
        st.header("👤 Create Profile")

        movie_options = movies[['movieId', 'title']]

        selected_movies = st.multiselect(
            "Select movies",
            options=movie_options['movieId'],
            format_func=lambda x: movie_options[movie_options['movieId']==x]['title'].values[0]
        )

        ratings_input = []
        for m in selected_movies:
            r = st.slider(f"Rate {movies[movies['movieId']==m]['title'].values[0]}", 1, 5, 3)
            ratings_input.append(r)

        if st.button("Submit Preferences"):
            new_user = add_user(selected_movies, ratings_input)

            updated = pd.read_csv("ratings.csv")
            ei, ew, um, mm, nnodes = build_graph(updated)

            new_model = GNNRecommender(nnodes)
            opt = torch.optim.Adam(new_model.parameters(), lr=0.01)

            for _ in range(50):
                new_model.train()
                opt.zero_grad()

                emb = new_model(ei, ew)
                src, dst = ei

                pos = (emb[src] * emb[dst]).sum(dim=1)
                neg_dst = torch.randint(0, nnodes, dst.size())
                neg = (emb[src] * emb[neg_dst]).sum(dim=1)

                loss = -torch.log(torch.sigmoid(pos)).mean() - torch.log(1 - torch.sigmoid(neg)).mean()

                loss.backward()
                opt.step()

            recs, sim_users = personalized_recommend(new_user, new_model, ei, ew, um)

            st.success(f"User ID: {new_user}")
            st.write("Similar Users:", sim_users)
            st.table(recs)

            if st.button("🔍 Show My Graph"):
                show_graph(new_user, sim_users)

            if st.button("🎯 My Focus Graph"):
                show_focus_graph(new_user)

# ---------------- ADMIN ----------------
elif menu == "Admin Login":
    st.header("🛠️ Admin Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login Admin"):
        if username == "admin" and password == "admin123":
            st.session_state.admin = True
            st.success("Admin Logged In")
        else:
            st.error("Wrong credentials")

    if st.session_state.get("admin"):
        st.header("🛠️ Admin Panel")

        uploaded = st.file_uploader("Upload New ratings.csv", type=["csv"])

        if uploaded:
            df = pd.read_csv(uploaded)
            df.to_csv("ratings.csv", index=False)
            st.success("Dataset updated")

        st.dataframe(ratings.head())
    
