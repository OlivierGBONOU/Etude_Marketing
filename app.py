import streamlit as st
import os
from PIL import Image

# Configuration de la page
st.set_page_config(
    page_title="Services de ménage - ENSEA",
    page_icon="🧹",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS personnalisé
def add_bg_and_styling():
    st.markdown("""
    <style>
        body {
            color: #2c3e50;
            background-color: #f8f9fa;
        }
        
        .stApp {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        h1 {
            color: #6c9cb0;
            font-family: 'Georgia', serif;
            font-weight: 600;
            padding-bottom: 1rem;
            border-bottom: 2px solid #d1e3eb;
        }
        
        h3 {
            color: #6c9cb0;
            font-family: 'Georgia', serif;
            margin-top: 1.5rem;
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid #d1e3eb;
        }
        
        .subtitle {
            font-style: italic;
            color: #2c3e50;
            margin-bottom: 1.5rem;
            font-size: 1.2rem;
        }
        
        .sommaire {
            background-color: #e8f4f8;
            border-left: 4px solid #6c9cb0;
            padding: 1.5rem;
            border-radius: 0 10px 10px 0;
            margin-bottom: 1.5rem;
        }
        
        .team {
            background-color: #e8f4f8;
            padding: 1.5rem;
            border-radius: 10px;
            margin-bottom: 1.5rem;
        }
        
        hr {
            height: 2px;
            background-color: #d1e3eb;
            border: none;
            margin: 2rem 0;
        }
        
        .image-container img {
            border-radius: 10px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
        }
        
        .stButton > button {
            background-color: #6c9cb0;
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            font-weight: 600;
            border-radius: 50px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #5a8697;
            transform: translateY(-2px);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        
        .cta-section {
            text-align: center;
            padding: 2rem;
            background-color: #f8f9fa;
            border-radius: 10px;
            margin-top: 1rem;
            border: 1px solid #d1e3eb;
        }
        
        .footer {
            text-align: center;
            padding-top: 1rem;
            font-size: 0.9rem;
            color: #6c757d;
        }
        
        .team-member {
            display: flex;
            align-items: center;
            margin-bottom: 1rem;
        }
        
        .team-circle {
            width: 20px;
            height: 20px;
            background-color: #6c9cb0;
            border-radius: 50%;
            margin-right: 15px;
            display: inline-block;
        }
        
        .styled-list {
            list-style-type: none;
            padding-left: 1rem;
        }
        
        .styled-list li {
            margin-bottom: 0.5rem;
            position: relative;
            padding-left: 1rem;
        }
        
        .styled-list li:before {
            content: "•";
            color: #6c9cb0;
            position: absolute;
            left: 0;
        }
    </style>
    """, unsafe_allow_html=True)

# Appliquer le style
add_bg_and_styling()

# En-tête
st.markdown("<h1>Services de ménage pour les étudiants de l'ENSEA</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Étude sur la pertinence de proposer des services de ménage adaptés aux besoins spécifiques des étudiants</p>", unsafe_allow_html=True)

# Contenu principal en colonnes
col1, col2 = st.columns([3, 2])

with col1:
    st.markdown("<div class='image-container'>", unsafe_allow_html=True)
    try:
        image_path = os.path.join(os.path.dirname(__file__), "menage.jpg")
        image = Image.open(image_path)
        st.image(image, caption="Services de ménage pour les étudiants", use_container_width=True)
    except:
        st.image("https://via.placeholder.com/800x400.png?text=Services+de+menage", 
                caption="Exemple d'image", 
                use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    # Dans votre code existant, remplacez la section du sommaire par ceci :
    st.markdown('## Sommaire', unsafe_allow_html=True)
    st.markdown("""
    <p class="content-text">
    1. Déroulement du terrain<br>
    2. Principaux résultats<br>
       &nbsp;&nbsp;2.1 Analyse du marché et de la demande<br>
       &nbsp;&nbsp;2.2 Analyse des besoins spécifiques<br>
       &nbsp;&nbsp;2.3 Modélisation de l’offre<br>
       &nbsp;&nbsp;2.4 Planification opérationnelle<br>
    3. Conclusion et recommandations
    </p>
    """, unsafe_allow_html=True)
    
    st.markdown("<h3>Notre équipe</h3>", unsafe_allow_html=True)
    st.markdown("""
    <div class="team">
        <div class="team-member">
            <div class="team-circle"></div>
            <span>AGBOGLA Komlan Richard</span>
        </div>
        <div class="team-member">
            <div class="team-circle"></div>
            <span>BANSIMBA Gautier</span>
        </div>
        <div class="team-member">
            <div class="team-circle"></div>
            <span>GBONOU Kossi Olivier Richard</span>
        </div>
        <div class="team-member">
            <div class="team-circle"></div>
            <span>PIERRE Jeff</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)

# Section CTA
st.markdown("""
<div class="cta-section">
    <h3>Prêt à explorer notre étude complète?</h3>
    <p style="margin-bottom: 2rem;">
        Découvrez notre analyse détaillée et nos recommandations pour la mise en place de services de ménage adaptés aux étudiants de l'ENSEA.
    </p>
</div>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <p>© 2025 - Étude réalisée dans le cadre du cours de méthodologie d'enquête - ENSEA</p>
</div>
""", unsafe_allow_html=True)