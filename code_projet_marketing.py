import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import pydotplus
import os
from io import StringIO
import openpyxl

# Configuration de la page
st.set_page_config(
    page_title="Services de ménage pour étudiants ENSEA",
    page_icon="🧹",
    layout="wide"
)

# Titre principal
st.title("Rapport du projet marketing : Services de ménage pour étudiants ENSEA")
st.markdown("*Étude sur la pertinence de proposer des services de ménage aux étudiants de l'ENSEA*")

# Chargement des données (à remplacer par votre chargement réel)
@st.cache_data
def load_data():
    script_dir = os.path.split(__file__)[0]
    
    # Construire le chemin complet vers le fichier Excel
    file_path = os.path.join(script_dir, "base_finale.xlsx")
    
    # Chargement du fichier
    df = pd.read_excel(file_path, engine="openpyxl")
    
    # Suppression de la colonne inutile
    if "Nationalite.1" in df.columns:
        df = df.drop(columns="Nationalite.1")

    # Remplacement des valeurs textuelles
    df["annee_etudes"] = df["annee_etudes"].replace({
        "1ère année": "1A",
        "2ème année": "2A",
        "3ème année": "3A"
    })

    df["sexe"] = df["sexe"].replace({
        "Homme": "H",
        "Femme": "F"
    })

    df['temps_menage_hebdo'] = df['temps_menage_hebdo'].replace({
        "Moins de 30 minutes": "<30min",
        "Entre 30 minutes et 1 heure": "30min-1h",
        "Entre 1 et 2 heures": "1-2h",
        "Plus de 2 heures": ">2h"
    })

    df['budget_mensuel'] = df['budget_mensuel'].replace({
        "Moins de 5000": "<5000",
        "Entre 5000 et 10000": "5000-10000",
        "Entre 15000 et 20000": "15000-20000"
    })

    # Mapping pour conversion numérique
    temps_map = {'<30min': 0.5, '30min-1h': 1.5, '1-2h': 2.5, '>2h': 3.5}
    budget_map = {'<5000': 3000, '5000-10000': 7500, '15000-20000': 17500}

    # Vérifier si les colonnes existent avant la conversion
    if 'temps_menage_hebdo' in df.columns:
        df['temps_menage_hebdo_num'] = df['temps_menage_hebdo'].map(temps_map)

    if 'budget_mensuel' in df.columns:
        df['budget_mensuel_num'] = df['budget_mensuel'].map(budget_map)

    return df

# Exécuter la fonction et stocker les données
df = load_data()

# Fonction pour convertir les colonnes binaires en format exploitable pour les visualisations
def prepare_binary_columns(df, prefix):
    binary_cols = [col for col in df.columns if col.startswith(prefix)]
    tasks = [col.replace(prefix, '') for col in binary_cols]
    
    # Créer un DataFrame pour les visualisations
    result_df = pd.DataFrame()
    for i, col in enumerate(binary_cols):
        result_df[tasks[i]] = df[col]
    
    return result_df, tasks

ordre_modalites = [
    'Pas du tout intéressé(e)', 
    'Plutôt pas intéressé(e)', 
    'Indécis(e)', 
    'Plutôt intéressé(e)', 
    'Très intéressé(e)'
]

# ----------------------- SECTION 1: ANALYSE DU MARCHÉ ET DE LA DEMANDE -----------------------
st.header("1. Analyse du marché et de la demande")

# Répartition par année d'études, type de logement et nombre d'occupants
st.subheader("Répartition démographique")

col1, col2 = st.columns([3, 1])

with col1:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Par sexe", "Par année d'études", "Par type de logement", "Par nombre d'occupants", "Par nationalité"])
    
    with tab1:
        # Calculer les valeurs et réinitialiser l'index
        sexe_counts = df['sexe'].value_counts().reset_index()
        sexe_counts.columns = ['sexe', 'count']
        
        fig = px.pie(sexe_counts, 
                    names='sexe', 
                    values='count', 
                    labels={'sexe': "Sexe", 'count': "Nombre d'étudiants"},
                    title="Répartition par sexe",
                    hole=0.4)  # Ajout du trou central pour créer un donut
        
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        # Calculer les valeurs et réinitialiser l'index
        annee_counts = df['annee_etudes'].value_counts().reset_index()
        
        fig = px.bar(annee_counts, 
                     x='annee_etudes', 
                     y='count', 
                     labels={'annee_etudes': "Année d'études", 'count': "Nombre d'étudiants"},
                     color='annee_etudes')
        fig.update_layout(title="Répartition par année d'études")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Calculer les valeurs et réinitialiser l'index
        type_logement_counts = df['type_logement'].value_counts().reset_index()
        
        fig = px.bar(type_logement_counts, 
                     x='type_logement', 
                     y='count', 
                     labels={'type_logement': "Type de logement", 'count': "Nombre d'étudiants"},
                     color='type_logement')
        fig.update_layout(title="Répartition par type de logement")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        # Calculer les valeurs et réinitialiser l'index
        nb_occupants_counts = df['nb_occupants'].value_counts().reset_index()
        
        fig = px.bar(nb_occupants_counts, 
                     x='nb_occupants', 
                     y='count', 
                     labels={'nb_occupants': "Nombre d'occupants", 'count': "Nombre d'étudiants"},
                     color='nb_occupants')
        fig.update_layout(title="Répartition par nombre d'occupants")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab5:
        # Calculer les valeurs et réinitialiser l'index
        nb_occupants_counts = df['Nationalite'].value_counts().reset_index()
        
        fig = px.bar(nb_occupants_counts, 
                     x='Nationalite', 
                     y='count', 
                     labels={'Nationalite': "Nationalité", 'count': "Nombre d'étudiants"},
                     color='Nationalite')
        fig.update_layout(title="Répartition par nombre d'occupants")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Démographie cible**
    
    La répartition démographique montre les segments les plus importants, permettant d'identifier les groupes prioritaires pour le développement du service de ménage.
    """)

# Évaluation de l'intérêt global pour le service
st.subheader("Évaluation de l'intérêt pour le service")

col1, col2 = st.columns([3, 1])

with col1:
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Intérêt global", "Par sexe", "Par type de logement", "Par année d'études", "Par nationalité"])
    
    with tab1:
        # Graphique circulaire pour "interet_service"
        fig = px.pie(df, names='interet_service', 
                     title="Intérêt global pour le service de ménage",
                     color='interet_service', 
                     category_orders={'interet_service': ordre_modalites},
                     color_discrete_map={'Pas du tout intéressé(e)': 'darkred',  'Plutôt pas intéressé(e)': 'orangered', 'Indécis(e)': 'gold', 'Plutôt intéressé(e)': 'lightgreen', 'Très intéressé(e)': 'green'})
        st.plotly_chart(fig, use_container_width=True)
        
    with tab2:
        # Graphique à mosaïque croisant "sexe" et "interet_service" avec effectifs bruts
        # Création de la table de contingence
        crosstab_counts = pd.crosstab(df['sexe'], df['interet_service'])
        crosstab_long_counts = crosstab_counts.reset_index().melt(id_vars='sexe', value_name='effectif')
        
        fig = px.bar(crosstab_long_counts, x='sexe', y='effectif', color='interet_service',
                    labels={'sexe': 'Sexe', 'effectif': "Nombre d'étudiants"},
                    title="Intérêt pour le service selon le sexe (effectifs bruts)",
                    category_orders={'interet_service': ordre_modalites},
                    color_discrete_map={'Pas du tout intéressé(e)': 'darkred',  'Plutôt pas intéressé(e)': 'orangered', 'Indécis(e)': 'gold', 'Plutôt intéressé(e)': 'lightgreen', 'Très intéressé(e)': 'green'})
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Graphique à mosaïque croisant "type_logement" et "interet_service" avec effectifs bruts
        # Création de la table de contingence
        crosstab_counts = pd.crosstab(df['type_logement'], df['interet_service'])
        crosstab_long_counts = crosstab_counts.reset_index().melt(id_vars='type_logement', value_name='effectif')
        
        fig = px.bar(crosstab_long_counts, x='type_logement', y='effectif', color='interet_service',
                    labels={'type_logement': 'Type de logement', 'effectif': "Nombre d'étudiants"},
                    title="Intérêt pour le service selon le type de logement (effectifs bruts)",
                    category_orders={'interet_service': ordre_modalites},
                    color_discrete_map={'Pas du tout intéressé(e)': 'darkred',  'Plutôt pas intéressé(e)': 'orangered', 'Indécis(e)': 'gold', 'Plutôt intéressé(e)': 'lightgreen', 'Très intéressé(e)': 'green'})
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab4:
        # Graphique à mosaïque croisant "annee_etudes" et "interet_service" avec effectifs bruts
        # Création de la table de contingence
        crosstab_counts = pd.crosstab(df['annee_etudes'], df['interet_service'])
        crosstab_long_counts = crosstab_counts.reset_index().melt(id_vars='annee_etudes', value_name='effectif')
        
        fig = px.bar(crosstab_long_counts, x='annee_etudes', y='effectif', color='interet_service',
                    labels={'annee_etudes': "Année d'études", 'effectif': "Nombre d'étudiants"},
                    title="Intérêt pour le service selon l'année d'études (effectifs bruts)",
                    category_orders={'interet_service': ordre_modalites},
                    color_discrete_map={'Pas du tout intéressé(e)': 'darkred',  'Plutôt pas intéressé(e)': 'orangered', 'Indécis(e)': 'gold', 'Plutôt intéressé(e)': 'lightgreen', 'Très intéressé(e)': 'green'})
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)
        
    with tab5:
        # Graphique à mosaïque croisant "Nationalite" et "interet_service"
        # Créons une table de contingence
        crosstab = pd.crosstab(df['Nationalite'], df['interet_service'], normalize='index')
        crosstab_long = crosstab.reset_index().melt(id_vars='Nationalite', value_name='proportion')
        
        fig = px.bar(crosstab_long, x='Nationalite', y='proportion', color='interet_service',
                     labels={'Nationalite': 'Nationalité', 'proportion': 'Proportion'},
                     title="Intérêt pour le service selon la nationalité",
                     category_orders={'interet_service': ordre_modalites},
                     color_discrete_map={'Pas du tout intéressé(e)': 'darkred',  'Plutôt pas intéressé(e)': 'orangered', 'Indécis(e)': 'gold', 'Plutôt intéressé(e)': 'lightgreen', 'Très intéressé(e)': 'green'})
        fig.update_layout(barmode='stack')
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Analyse de l'intérêt**
    
    L'intérêt pour le service varie significativement selon le type de logement et l'année d'études. Cette analyse permet d'identifier les segments les plus réceptifs et de cibler les efforts marketing en conséquence.
    """)

# Analyse des freins et motivations
st.subheader("Analyse des freins et motivations")

col1, col2 = st.columns([3, 1])

with col1:
    # Diagramme en barres horizontales pour "raison_non_interet"
    raison_non_df = df[df['interet_service'] == 'Pas du tout intéressé(e)']['raison_non_interet'].value_counts().reset_index()
    fig = px.bar(raison_non_df, y='raison_non_interet', x='count', 
                 title="Raisons du désintérêt pour le service",
                 labels={'raison_non_interet': 'Raison', 'count': 'Nombre d\'étudiants'},
                 orientation='h')
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    st.plotly_chart(fig, use_container_width=True)
    
    seuil_pourcentage=4
    
    # Préparation des données
    periode_count = df.groupby(['periode_difficile', 'principaux_frein']).size().reset_index(name='count')
    frein_interet = df.groupby(['principaux_frein', 'interet_service']).size().reset_index(name='count')
    
    # Filtrer les connexions pour ne garder que celles dépassant un certain seuil
    total_periode_frein = periode_count['count'].sum()
    total_frein_interet = frein_interet['count'].sum()
    
    # Calculer le seuil absolu basé sur le pourcentage
    seuil_periode_frein = (seuil_pourcentage / 100) * total_periode_frein
    seuil_frein_interet = (seuil_pourcentage / 100) * total_frein_interet
    
    # Filtrer les connexions significatives
    periode_count_filtered = periode_count[periode_count['count'] >= seuil_periode_frein]
    frein_interet_filtered = frein_interet[frein_interet['count'] >= seuil_frein_interet]
    
    # Création de la liste des nœuds (uniquement ceux qui restent après filtrage)
    nodes = list(set(periode_count_filtered['periode_difficile'].tolist() + 
                    periode_count_filtered['principaux_frein'].tolist() + 
                    frein_interet_filtered['interet_service'].tolist()))
    
    # Création d'un mapping pour les indices des nœuds
    node_indices = {node: i for i, node in enumerate(nodes)}
    
    # Création des liens
    links_periode_frein = [dict(source=node_indices.get(row['periode_difficile']), 
                               target=node_indices.get(row['principaux_frein']), 
                               value=row['count']) 
                         for _, row in periode_count_filtered.iterrows() 
                         if row['periode_difficile'] in node_indices and row['principaux_frein'] in node_indices]
    
    links_frein_interet = [dict(source=node_indices.get(row['principaux_frein']), 
                               target=node_indices.get(row['interet_service']), 
                               value=row['count']) 
                         for _, row in frein_interet_filtered.iterrows()
                         if row['principaux_frein'] in node_indices and row['interet_service'] in node_indices]
    
    links = links_periode_frein + links_frein_interet
    
    # Création du diagramme Sankey
    # Couleur par défaut (bleu clair) et couleur au survol (bleu foncé)
    default_color = 'rgba(173, 216, 230, 0.8)'  # Bleu clair
    hover_color = 'rgba(0, 0, 139, 0.8)'  # Bleu foncé

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=15,
            thickness=20,
            line=dict(color="black", width=0.5),
            label=nodes        
        ),
        link=dict(
            source=[link['source'] for link in links],
            target=[link['target'] for link in links],
            value=[link['value'] for link in links],
            color=[default_color] * len(links),
            hoverinfo="all",
            customdata=[hover_color] * len(links)  # Couleur au survol
        )
    )])
    
    fig.update_layout(
        title_text="Parcours simplifié: période difficile → freins → intérêt pour le service",
        height=600,
        font=dict(size=12)
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Analyse des freins**
    
    L'identification des principaux freins nous permet de comprendre pourquoi certains étudiants ne sont pas intéressés par le service. Le diagramme Sankey illustre comment les périodes difficiles et les freins influencent l'intérêt pour le service.
    """)
    

    # Profil de la population étudiante
st.subheader("Profil de la population estudiantine")

# Clustering pour personas
# Sélection des variables pour le clustering
cluster_vars = ['type_logement', 'nb_occupants', 'temps_menage_hebdo', 'importance_proprete']

# Préparation des données pour le clustering (encodage one-hot pour variables catégorielles)
cluster_data = pd.get_dummies(df[cluster_vars])
cluster_data = cluster_data.drop("importance_proprete_Très importante", axis=1)

# Standardisation
scaler = StandardScaler()
cluster_data_scaled = scaler.fit_transform(cluster_data)

# Clustering K-means
kmeans = KMeans(n_clusters=3, random_state=42)
df['cluster'] = kmeans.fit_predict(cluster_data_scaled)

# Visualisation des clusters avec PCA
pca = PCA()  # Pas de limitation du nombre de composantes pour l'éboulis
pca_full = pca.fit_transform(cluster_data_scaled)

# Garder uniquement les 2 premières composantes pour la visualisation
pca_result = pca_full[:, :2]

# Récupération des valeurs propres
explained_variance = pca.explained_variance_
explained_variance_ratio = pca.explained_variance_ratio_

# Création des sous-graphiques: un pour l'éboulis des valeurs propres et un pour le cercle de corrélation
fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Graphique 1: Histogramme des valeurs propres
n_components = min(10, len(explained_variance))  # Limiter à 10 ou moins si moins de variables
x_range = range(1, n_components + 1)

# Créer l'histogramme
ax1.bar(x_range, explained_variance[:n_components], width=0.8, align='center')
ax1.set_title('Histogramme des valeurs propres')
ax1.set_xlabel('Composante principale')
ax1.set_ylabel('Valeur propre')
ax1.grid(True)

# Ajouter le pourcentage de variance expliquée au-dessus des barres
for i, ratio in enumerate(explained_variance_ratio[:n_components]):
    ax1.text(i + 1, explained_variance[i], f'{ratio:.1%}', 
             va='bottom', ha='center', fontsize=8)

# Affichage de la variance cumulée
cum_variance_ratio = np.cumsum(explained_variance_ratio[:n_components])
ax1_twin = ax1.twinx()
ax1_twin.plot(x_range, cum_variance_ratio, 'r-', marker='s', linewidth=2)
ax1_twin.set_ylabel('Variance cumulée expliquée', color='r')
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1_twin.grid(False)

# Graphique 2: Cercle de corrélation
# Utiliser uniquement les 2 premières composantes pour le cercle de corrélation
pca_components = pca.components_[:2, :]
feature_names = cluster_data.columns

# Tracer le cercle de corrélation
circle = plt.Circle((0, 0), 1, fill=False, color='gray', linestyle='--')
ax2.add_patch(circle)

# Définir un seuil de corrélation (ne montrer que les variables fortement corrélées)
correlation_threshold = 0.38

# Tracer les flèches pour chaque variable qui dépasse le seuil
for i, (x, y) in enumerate(zip(pca_components[0, :], pca_components[1, :])):
    # Calculer la longueur du vecteur (force de la corrélation)
    length = np.sqrt(x**2 + y**2)
    
    # N'afficher que les variables qui dépassent le seuil
    if length > correlation_threshold:
        ax2.arrow(0, 0, x, y, head_width=0.05, head_length=0.05, fc='blue', ec='blue')
        ax2.text(x * 1.1, y * 1.1, feature_names[i], color='black', ha='center', va='center', fontweight='bold')
    # Optionnel : afficher en gris clair les variables sous le seuil
    else:
        ax2.arrow(0, 0, x, y, head_width=0.03, head_length=0.03, fc='lightgray', ec='lightgray', alpha=0.5)
        ax2.text(x * 1.1, y * 1.1, feature_names[i], color='gray', ha='center', va='center', alpha=0.5)

ax2.set_xlim(-1.2, 1.2)
ax2.set_ylim(-1.2, 1.2)
ax2.grid(True)
ax2.set_title('Cercle de corrélation')
ax2.set_xlabel('Composante principale 1')
ax2.set_ylabel('Composante principale 2')

# Ajout d'une ligne horizontale et verticale pour aider à l'interprétation
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
ax2.axvline(x=0, color='gray', linestyle='-', alpha=0.3)

plt.tight_layout()
st.pyplot(fig1)

# Graphique 3: Segmentation des étudiants (maintenant en bas)
fig2, ax3 = plt.subplots(figsize=(9, 5))
scatter = ax3.scatter(pca_result[:, 0], pca_result[:, 1], c=df['cluster'], cmap='viridis')
ax3.set_title('Segmentation des étudiants par clustering (3 personas)')
ax3.set_xlabel('Composante principale 1')
ax3.set_ylabel('Composante principale 2')
plt.tight_layout()
st.pyplot(fig2)

# Colonne d'explication
st.markdown("""
**Analyse des personas et de l'ACP**

L'éboulis des valeurs propres montre l'importance relative de chaque composante principale et aide à déterminer le nombre optimal de dimensions à conserver.

**Interprétation du cercle de corrélation :**

Le cercle de corrélation montre comment les variables originales sont liées aux deux composantes principales. Les variables en **gras** sont celles qui sont fortement corrélées (seuil > 0.5) avec les composantes principales et donc les plus influentes dans la formation des clusters. Les variables en gris clair sont moins déterminantes.

**Segmentation des étudiants :**

Le clustering révèle 3 profils distincts d'étudiants avec des besoins et comportements différents face au ménage. Ces personas peuvent guider la personnalisation des offres de services.
""")
# ----------------------- SECTION 2: ANALYSE DES BESOINS SPÉCIFIQUES -----------------------
st.header("2. Analyse des besoins spécifiques")

# Charge de ménage actuelle
st.subheader("Charge de ménage actuelle")

col1, col2 = st.columns([3, 1])

with col1:
    # Boîtes à moustaches comparant "temps_menage_hebdo" selon le type de logement
    contingency_table = pd.crosstab(df['temps_menage_hebdo_num'], df['type_logement'])

    # Création du heatmap
    fig = px.imshow(contingency_table,
                labels=dict(x="Type de logement", y="Temps hebdomadaire", color="Fréquence"),
                title="Temps consacré au ménage selon le type de logement",
                color_continuous_scale="Blues",
                y=contingency_table.index)
    st.plotly_chart(fig, use_container_width=True)
    
    # Scatter plot croisant "difficulte_etudes_menage" et "temps_menage_hebdo"
    difficulte_map = {'Non, jamais': 0, 'Rarement': 1, 'Oui, parfois': 2, "Oui, très souvent": 3}
    df['difficulte_num'] = df['difficulte_etudes_menage'].map(difficulte_map)
    
    fig = px.scatter(df, x='temps_menage_hebdo_num', y='difficulte_num', 
                     color='type_logement',
                     size='temps_menage_hebdo_num',
                     labels={'temps_menage_hebdo_num': 'Temps hebdomadaire (heures)', 
                             'difficulte_num': 'Difficulté à concilier études et ménage'},
                     title="Relation entre temps de ménage et difficulté à concilier avec les études",
                     category_orders={'difficulte_num': [0, 1, 2, 3]})
    fig.update_layout(yaxis=dict(tickmode='array', tickvals=[0, 1, 2, 3], ticktext=['Non, jamais', 'Rarement', 'Oui, parfois', 'Oui, très souvent']))
    st.plotly_chart(fig, use_container_width=True)
    
    # Heat map des corrélations
    # Sélection des variables numériques pour la matrice de corrélation
    corr_vars = ['temps_menage_hebdo_num', 'difficulte_num', 'tache_contraignante_sols', 
                 'tache_contraignante_sdb', 'tache_contraignante_cuisine', 'tache_contraignante_vaisselle',
                 'tache_contraignante_linge', 'tache_contraignante_rangement']
    
    corr_matrix = df[corr_vars].corr()
    
    fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                     title="Corrélations entre temps consacré au ménage et autres variables",
                     color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Analyse de la charge de ménage**
    
    Les étudiants consacrent un temps variable au ménage selon leur type de logement. On observe une corrélation entre le temps consacré au ménage et la difficulté à concilier avec les études, suggérant un réel besoin de services d'aide.
    """)

# Tâches problématiques
st.subheader("Tâches problématiques")

col1, col2 = st.columns([3, 1])

with col1:
    # Préparation des données pour le graphique en toile d'araignée
    taches_df, taches_list = prepare_binary_columns(df, 'tache_contraignante_')
    taches_df = pd.concat([taches_df, df['type_logement']], axis=1)
    
    # Calcul des moyennes par type de logement
    taches_by_logement = taches_df.groupby('type_logement').mean(numeric_only=True).reset_index()
    
    # Graphique en toile d'araignée
    fig = go.Figure()
    
    for logement in taches_by_logement['type_logement'].unique():
        row = taches_by_logement[taches_by_logement['type_logement'] == logement]
        values = row.iloc[0, 1:].tolist()
        # Ajouter la première valeur à la fin pour fermer le polygone
        values.append(values[0])
        categories = taches_list + [taches_list[0]]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=logement
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Tâches contraignantes par type de logement",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Analyse des tâches problématiques**
    
    Le graphique radar montre clairement quelles tâches sont les plus contraignantes selon le type de logement. Cette analyse permet d'adapter les offres de service aux besoins spécifiques de chaque segment.
    
    L'analyse factorielle révèle des clusters d'étudiants partageant des difficultés similaires, permettant une segmentation fine des besoins.
    """)

# Périodes critiques
st.subheader("Périodes critiques")

col1, col2 = st.columns([3, 1])

with col1:
    # Diagramme en barres des périodes difficiles
    periode_difficile_counts = df['periode_difficile'].value_counts().reset_index()
    
    fig = px.bar(periode_difficile_counts, 
                 x='periode_difficile', 
                 y='count', 
                 labels={'periode_difficile': 'Période', 'count': 'Nombre d\'étudiants'},
                 title="Périodes difficiles pour gérer le ménage",
                 color='periode_difficile')
    st.plotly_chart(fig, use_container_width=True)
    
with col2:
    st.markdown("""
    **Analyse des périodes critiques**
    
    Les examens et les périodes de projets sont les moments où les étudiants ont le plus de difficultés à gérer leur ménage. La visualisation de la charge académique tout au long de l'année permet d'identifier les périodes où le service serait le plus valorisé.
    """)

# ----------------------- SECTION 3: MODÉLISATION DE L'OFFRE -----------------------
st.header("3. Modélisation de l'offre")

# Tarification et budget
st.subheader("Tarification et budget")

col1, col2 = st.columns([3, 1])

with col1:
    # Histogramme du "budget_mensuel" avec ligne de distribution cumulative
    budget_counts = df['budget_mensuel'].value_counts().reset_index()
    budget_counts = budget_counts.sort_values(by='budget_mensuel', key=lambda x: x.map({'<20€': 1, '20-50€': 2, '50-80€': 3, '>80€': 4}))
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Histogramme
    fig.add_trace(
        go.Bar(x=budget_counts['budget_mensuel'], y=budget_counts['count'], name="Fréquence"),
        secondary_y=False,
    )
    
    fig.update_layout(
        title_text="Distribution du budget mensuel",
        xaxis_title="Budget mensuel",
    )
    
    fig.update_yaxes(title_text="Nombre d'étudiants", secondary_y=False)
    fig.update_yaxes(title_text="Distribution cumulative", secondary_y=True)
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Diagramme à barres groupées comparant le "budget_mensuel" selon différentes variables
    for var in ['nb_occupants', 'type_logement']:
        crosstab = pd.crosstab(df[var], df['budget_mensuel'])
        fig = px.bar(crosstab, 
                     barmode='group',
                     title=f"Budget mensuel selon {var}")
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Analyse du budget**
    
    La majorité des étudiants disposent d'un budget limité pour les services de ménage. Cette analyse permet d'établir une stratégie tarifaire adaptée aux différents segments. On note également que le budget varie selon le type de logement et le nombre d'occupants.
    """)

# Composition des services
st.subheader("Composition des services")

col1, col2 = st.columns([3, 1])

with col1:
    # Diagramme en radar comparant l'intérêt pour différentes prestations
    prestations_cols = ['prestation_complet', 'prestation_zones', 'prestation_repassage', 
                      'prestation_lessive', 'prestation_rangement']
    prestations_labels = ['Ménage complet', 'Zones spécifiques', 'Repassage', 'Lessive', 'Rangement']
    
    prestations_means = df[prestations_cols].mean().values.tolist()
    # Ajouter la première valeur à la fin pour fermer le polygone
    prestations_means.append(prestations_means[0])
    prestations_labels.append(prestations_labels[0])
    
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=prestations_means,
        theta=prestations_labels,
        fill='toself',
        name='Intérêt moyen'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        title="Intérêt pour les différentes prestations"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Analyse du panier moyen
    corr_prestations = df[prestations_cols].corr()
    
    fig = px.imshow(corr_prestations, text_auto=True, aspect="auto",
                     title="Corrélations entre les prestations demandées",
                     labels=dict(x="Prestation", y="Prestation", color="Corrélation"),
                     x=prestations_labels[:-1], y=prestations_labels[:-1],
                     color_continuous_scale='RdBu_r')
    st.plotly_chart(fig, use_container_width=True)
    
    # Matrice BCG adaptée
    # Calcul de l'attractivité (proportions d'étudiants intéressés)
    attractivite = df[prestations_cols].mean()
    
    # Calcul du potentiel de revenus (simulation)
    # Prix hypothétiques pour chaque service
    prix = {'prestation_complet': 30, 'prestation_zones': 15, 
            'prestation_repassage': 10, 'prestation_lessive': 8, 
            'prestation_rangement': 12}
    
    potentiel_revenus = {col: attractivite[col] * prix[col] for col in prestations_cols}
    potentiel_revenus = pd.Series(potentiel_revenus)
    
    # Création du dataframe pour la visualisation
    bcg_df = pd.DataFrame({
        'Attractivité': attractivite,
        'Potentiel de revenus': potentiel_revenus,
        'Prestation': prestations_labels[:-1]
    })
    
    fig = px.scatter(bcg_df, x='Attractivité', y='Potentiel de revenus', 
                     text='Prestation', size='Potentiel de revenus',
                     title="Matrice BCG adaptée des prestations",
                     labels={'Attractivité': 'Attractivité (proportion d\'intéressés)', 
                             'Potentiel de revenus': 'Potentiel de revenus'},
                     color='Prestation')
    
    # Ajouter des lignes pour diviser en quadrants
    fig.add_hline(y=potentiel_revenus.median(), line_dash="dash", line_color="gray")
    fig.add_vline(x=attractivite.median(), line_dash="dash", line_color="gray")
    
    # Ajouter des annotations pour les quadrants
    fig.add_annotation(x=0.9, y=0.9, text="Étoiles", showarrow=False, xref="paper", yref="paper")
    fig.add_annotation(x=0.1, y=0.9, text="Dilemmes", showarrow=False, xref="paper", yref="paper")
    fig.add_annotation(x=0.9, y=0.1, text="Vaches à lait", showarrow=False, xref="paper", yref="paper")
    fig.add_annotation(x=0.1, y=0.1, text="Poids morts", showarrow=False, xref="paper", yref="paper")
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Analyse des prestations**
    
    Le radar montre l'attractivité relative des différentes prestations. La matrice de corrélation identifie les services qui sont souvent demandés ensemble, permettant de créer des offres groupées efficaces.
    
    La matrice BCG adaptée permet d'identifier les services stratégiques à privilégier (étoiles) et ceux qui peuvent servir d'offres d'appel (vaches à lait).
    """)

# Modalités de service
st.subheader("Modalités de service")

col1, col2 = st.columns([3, 1])

with col1:
    # Diagramme en barres pour "frequence_utilisation" préférée
    frequence_counts = df['frequence_utilisation'].value_counts().reset_index()
    
    fig = px.bar(frequence_counts, 
                 x='frequence_utilisation', 
                 y='count', 
                 labels={'frequence_utilisation': 'Fréquence', 'count': 'Nombre d\'étudiants'},
                 title="Fréquence d'utilisation préférée",
                 color='frequence_utilisation')
    st.plotly_chart(fig, use_container_width=True)
    
    # Graphique comparatif "interet_abonnement" vs prestations ponctuelles
    fig = px.pie(df, names='interet_abonnement', 
                 title="Intérêt pour une formule d'abonnement",
                 color='interet_abonnement',
                 color_discrete_map={'Plutôt pas intéressé(e)':'yellow','Pas du tout intéressé(e)':'red','Très intéressé(e)':'green', 'Indécis(e)':'gold', 'Plutôt intéressé(e)':'crimson'})
    st.plotly_chart(fig, use_container_width=True)
    
    # Diagrammes circulaires pour "presence_menage" et "confiance_cles"
    col_a, col_b = st.columns(2)
    
    with col_a:
        fig = px.pie(df, names='presence_menage', 
                     title="Préférence pour la présence pendant le ménage",
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    
    with col_b:
        fig = px.pie(df, names='confiance_cles', 
                     title="Disposition à confier les clés",
                     hole=0.4,
                     color_discrete_map={'Non, je préfère être présent(e)':'red', 'Oui, mais avec des réserves':'crimson', 'Je ne sais pas':'green', 'Oui, sans problème':'gold'})
        st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Analyse des modalités**
    
    Les étudiants montrent des préférences diverses en termes de fréquence d'utilisation, avec une tendance vers des services réguliers. L'intérêt pour les abonnements est significatif, suggérant un potentiel pour des offres de fidélisation.
    
    Les questions de présence et de confiance des clés sont cruciales pour le modèle opérationnel du service.
    """)

# ----------------------- SECTION 4: PLANIFICATION OPÉRATIONNELLE -----------------------
st.header("4. Planification opérationnelle")

# Disponibilités et préférences horaires
st.subheader("Disponibilités et préférences horaires")

col1, col2 = st.columns([3, 1])

with col1:
    # Heat map des jours préférés
    jour_cols = ['jour_lundi', 'jour_mardi', 'jour_mercredi', 
                'jour_jeudi', 'jour_vendredi', 'jour_samedi', 'jour_dimanche']
    jour_labels = ['Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi', 'Dimanche']
    
    # Agrégation par plage horaire et jour
    jour_plage = df.groupby('plage_horaire')[jour_cols].mean()
    
    # Renommer les colonnes
    jour_plage.columns = jour_labels
    
    fig = px.imshow(jour_plage, text_auto=True, aspect="auto",
                     title="Disponibilités selon les jours et plages horaires",
                     labels=dict(x="Jour", y="Plage horaire", color="Proportion"),
                     color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_width=True)
    
    # Graphique en quadrants pour les créneaux optimaux
    # Création d'un score d'optimalité pour chaque jour/plage
    jours_optimaux = jour_plage.unstack().reset_index()
    jours_optimaux.columns = ['jour', 'plage', 'score']
    
    # Score sur l'axe des x: proportion d'étudiants disponibles
    # Score sur l'axe des y: facilité opérationnelle (simulée)
    np.random.seed(42)
    jours_optimaux['facilite_operationnelle'] = np.random.uniform(0.3, 0.9, len(jours_optimaux))
    
    fig = px.scatter(jours_optimaux, x='score', y='facilite_operationnelle', 
                     color='jour', symbol='plage', size='score',
                     labels={'score': 'Demande étudiante', 'facilite_operationnelle': 'Facilité opérationnelle'},
                     title="Créneaux optimaux pour les services")
    
    # Ajouter des lignes pour diviser en quadrants
    fig.add_hline(y=jours_optimaux['facilite_operationnelle'].median(), line_dash="dash", line_color="gray")
    fig.add_vline(x=jours_optimaux['score'].median(), line_dash="dash", line_color="gray")
    
    # Ajouter des annotations pour les quadrants
    fig.add_annotation(x=0.9, y=0.9, text="Créneaux optimaux", showarrow=False, xref="paper", yref="paper")
    fig.add_annotation(x=0.1, y=0.9, text="Faciles mais peu demandés", showarrow=False, xref="paper", yref="paper")
    fig.add_annotation(x=0.9, y=0.1, text="Demandés mais difficiles", showarrow=False, xref="paper", yref="paper")
    fig.add_annotation(x=0.1, y=0.1, text="À éviter", showarrow=False, xref="paper", yref="paper")
    
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("""
    **Analyse des disponibilités**
    
    La heat map révèle les plages horaires les plus demandées, permettant d'optimiser la planification des services. Le graphique en quadrants identifie les créneaux qui combinent une forte demande étudiante et une facilité opérationnelle.
    """)

# ----------------------- CONCLUSION -----------------------
st.header("Conclusion")
st.markdown("""
""")

# ----------------------- SIDEBAR FOR FILTERS -----------------------
with st.sidebar:
    st.header("Filtres")
    st.markdown("Sélectionnez les filtres pour affiner l'analyse")
    
    # Ajout de filtres pour l'analyse
    annee_filter = st.multiselect("Année d'études", df['annee_etudes'].unique(), default=df['annee_etudes'].unique())
    logement_filter = st.multiselect("Type de logement", df['type_logement'].unique(), default=df['type_logement'].unique())
    interet_filter = st.multiselect("Intérêt pour le service", df['interet_service'].unique(), default=df['interet_service'].unique())
    
    st.markdown("---")
    st.markdown("**Note:** .")