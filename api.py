# api.py — Backend Flask pour Martengo Prediction
# Déploiement : Railway / Render / Heroku

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from io import StringIO
import os
import re
import pickle
import threading
import requests as http_requests
from sklearn.ensemble import HistGradientBoostingClassifier

app = Flask(__name__)
CORS(app)

# ============================================================
# DONNÉES HISTORIQUES
# ============================================================
# historique_notes.csv : versionné dans le repo (117k lignes, source de vérité)
# historique_courses.csv : courses ajoutées via /ajouter
#
# PERSISTANCE :
#   - Si DATABASE_URL est défini (PostgreSQL sur Render/Railway) → stockage SQL
#   - Sinon → fallback CSV local (éphémère sur Render Free, acceptable en dev)
#
# Pour activer Postgres : ajoutez DATABASE_URL dans les variables d'env Render.
# La table est créée automatiquement au démarrage.
# ============================================================
HISTORIQUE_PATH      = "historique_notes.csv"
CSV_PATH        = "historique_courses.csv"

# ── Connexion PostgreSQL optionnelle ──────────────────────────
_pg_conn = None

def _get_pg():
    """Retourne une connexion PostgreSQL si DATABASE_URL est défini, sinon None."""
    global _pg_conn
    if _pg_conn is not None:
        try:
            _pg_conn.cursor().execute("SELECT 1")
            return _pg_conn
        except Exception:
            _pg_conn = None
    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        return None
    try:
        import psycopg2
        # Render fournit parfois "postgres://" → psycopg2 veut "postgresql://"
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        _pg_conn = psycopg2.connect(db_url, sslmode="require")
        _pg_conn.autocommit = True
        # Créer la table si elle n'existe pas
        with _pg_conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS courses_manuelles (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    numero INTEGER NOT NULL,
                    note FLOAT,
                    rapport FLOAT,
                    rang_arrivee INTEGER,
                    score_cible INTEGER DEFAULT 0
                )
            """)
        print("✅ PostgreSQL connecté et table prête")
        return _pg_conn
    except Exception as e:
        print(f"⚠️  PostgreSQL indisponible ({e}) — fallback CSV")
        return None


def _lire_courses_manuelles():
    """Lit les courses ajoutées manuellement (PG ou CSV)."""
    conn = _get_pg()
    if conn:
        try:
            return pd.read_sql("SELECT date, numero, note, rapport, rang_arrivee, score_cible FROM courses_manuelles", conn)
        except Exception as e:
            print(f"⚠️  Lecture PG échouée ({e}) — fallback CSV")
    if os.path.exists(CSV_PATH):
        return pd.read_csv(CSV_PATH)
    return pd.DataFrame(columns=['date','numero','note','rapport','rang_arrivee','score_cible'])


def _ecrire_courses_manuelles(df_new_rows):
    """Persiste de nouvelles lignes (PG prioritaire, sinon CSV)."""
    conn = _get_pg()
    if conn:
        try:
            from psycopg2.extras import execute_values
            with conn.cursor() as cur:
                execute_values(cur,
                    "INSERT INTO courses_manuelles (date, numero, note, rapport, rang_arrivee, score_cible) VALUES %s",
                    [(row['date'], int(row['numero']), float(row['note']),
                      float(row['rapport']), int(row['rang_arrivee']), int(row.get('score_cible', 0)))
                     for _, row in df_new_rows.iterrows()]
                )
            print(f"✅ {len(df_new_rows)} lignes écrites en PostgreSQL")
            return
        except Exception as e:
            print(f"⚠️  Écriture PG échouée ({e}) — fallback CSV")
    # Fallback CSV (éphémère sur Render Free)
    df_all = _lire_courses_manuelles()
    df_all = pd.concat([df_all, df_new_rows], ignore_index=True).drop_duplicates()
    df_all.to_csv(CSV_PATH, index=False)
    print(f"⚠️  Données écrites dans {CSV_PATH} (éphémère — seront perdues au redémarrage)")


FEATURES = [
    'note','rapport','log_rapport',
    'note_normalisee','inverse_rapport',
    'score_valeur','rapport_over_10','valeur_brute'
]
FEATURES_ABSOLU = FEATURES + ['ratio_note_rapport']

# ============================================================
# LOGIQUE ML (identique aux blocs 3 & 5)
# ============================================================
def _enrichir(df_in, nm=None, ns=None):
    d = df_in.copy()
    d['log_rapport']   = np.log1p(d['rapport'])
    if nm is None:
        nm = d['note'].mean()
        ns = d['note'].std(); ns = ns if ns != 0 else 1.0
    d['note_normalisee']    = (d['note'] - nm) / ns
    d['inverse_rapport']    = 1.0 / (1.0 + d['rapport'])
    d['score_valeur']       = d['note'] / (1.0 + d['log_rapport'])
    d['rapport_over_10']    = np.maximum(0, d['rapport'] - 10)
    d['valeur_brute']       = d['note'] * d['log_rapport']
    d['ratio_note_rapport'] = d['note'] / (d['rapport'] + 1)
    return d, nm, ns


def _entrainer(df_source, features, target_col):
    d = df_source.sort_values('date')
    last = d['date'].max()
    train = d[d['date'] < last]
    clf = HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=6,
        max_iter=800, l2_regularization=0.5, random_state=42
    )
    clf.fit(train[features], train[target_col])
    return clf


def initialiser():
    global df, model, note_mean, note_std, modele_abs, note_mean_a, note_std_a

    # Charger historique complet (9575 courses)
    if os.path.exists(HISTORIQUE_PATH):
        df = pd.read_csv(HISTORIQUE_PATH)
        print(f"✅ Historique chargé : {len(df)} lignes / {df['date'].nunique()} dates")
    elif os.path.exists(CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        print(f"✅ CSV manuel chargé : {len(df)} lignes")
    else:
        print("⚠️  Aucun fichier historique trouvé — données vides")
        df = pd.DataFrame(columns=['date','r_num','c_num','numero','nom','note','rapport','rang_arrivee'])

    # Fusionner avec courses ajoutées manuellement (PG ou CSV)
    df_manual = _lire_courses_manuelles()
    if len(df_manual) > 0:
        common_cols = ['date', 'numero', 'note', 'rapport', 'rang_arrivee']
        df_manual = df_manual[[c for c in common_cols if c in df_manual.columns]]
        df_base   = df[[c for c in common_cols if c in df.columns]]
        df = pd.concat([df_base, df_manual], ignore_index=True).drop_duplicates()
        print(f"✅ {len(df_manual)} lignes manuelles fusionnées")

    # Si pas de données : skip l'entraînement (modèles v5 fallback uniquement)
    if len(df) == 0:
        print("⚠️  Pas de données historiques v5 — skip entraînement modèle fallback")
        model = None
        modele_abs = None
        note_mean = note_std = note_mean_a = note_std_a = 0.0
        return

    df['date'] = pd.to_datetime(df['date'])
    # Assurer colonne score_cible pour compatibilité
    if 'score_cible' not in df.columns:
        df['score_cible'] = 0

    try:
        # Modèle principal (cote >= 10)
        df_p = df.copy()
        df_p['target'] = ((df_p['rapport'] >= 10) & (df_p['rang_arrivee'] <= 3)).astype(int)
        df_p, note_mean, note_std = _enrichir(df_p)
        model = _entrainer(df_p, FEATURES, 'target')

        # Modèle absolu
        df_a = df.copy()
        df_a['target_absolu'] = (df_a['rang_arrivee'] <= 3).astype(int)
        df_a, note_mean_a, note_std_a = _enrichir(df_a)
        modele_abs = _entrainer(df_a, FEATURES_ABSOLU, 'target_absolu')

        print(f"✅ Modèles entraînés sur {len(df)} lignes / {df['date'].nunique()} courses")
    except Exception as e:
        print(f"⚠️  Impossible d'entraîner les modèles v5 fallback : {e}")
        model = None
        modele_abs = None
        note_mean = note_std = note_mean_a = note_std_a = 0.0


# ============================================================
# ROUTES API
# ============================================================
@app.route('/', methods=['GET', 'HEAD'])
def index():
    return jsonify({'status': 'ok', 'service': 'martengo-api'})


@app.route('/programme', methods=['GET'])
def programme():
    """Proxy vers l'API PMU — retourne le programme du jour."""
    date_str = request.args.get('date', '')
    if not date_str:
        from datetime import datetime
        date_str = datetime.now().strftime('%d%m%Y')
    try:
        url  = f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme/{date_str}"
        resp = http_requests.get(url, timeout=10)
        if resp.status_code != 200:
            return jsonify({"error": f"API PMU {resp.status_code}"}), resp.status_code
        data = resp.json()
        courses = []
        for reunion in data.get('programme', {}).get('reunions', []):
            r_num = reunion.get('numOfficiel') or reunion.get('numReunion')
            lieu  = (reunion.get('hippodrome') or {}).get('libelleCourt', '') or \
                    (reunion.get('hippodrome') or {}).get('libelleLong', '')
            for course in reunion.get('courses', []):
                c_num   = course.get('numOrdre') or course.get('numExterne')
                heure_ts = course.get('heureDepart', 0) or 0
                if heure_ts:
                    from datetime import datetime as dt, timedelta
                    heure = (dt.fromtimestamp(heure_ts/1000) + timedelta(hours=2)).strftime('%H:%M')
                else:
                    heure = '—'
                courses.append({
                    'r_num':   r_num,
                    'c_num':   c_num,
                    'disc':    course.get('discipline', ''),
                    'dist':    course.get('distance', 0) or 0,
                    'lieu':    lieu,
                    'heure':   heure,
                    'libelle': course.get('libelle', '') or course.get('libelleCourt', ''),
                })
        return jsonify({'date': date_str, 'courses': courses})
    except Exception as e:
        return jsonify({"error": str(e)}), 502


@app.route('/health', methods=['GET'])
def health():
    if df is not None and 'r_num' in df.columns and 'c_num' in df.columns:
        nb_courses = int(df.groupby(['date','r_num','c_num']).ngroups)
    elif df is not None:
        nb_courses = int(df['date'].nunique())
    else:
        nb_courses = 0
    return jsonify({"status": "ok", "courses": nb_courses})


@app.route('/predict', methods=['POST'])
def predict():
    """
    Body JSON : { "chevaux": [{"numero":1,"note":15,"rapport":12.5}, ...] }
    """
    data = request.get_json()
    chevaux = data.get('chevaux', [])
    if not chevaux:
        return jsonify({"error": "Aucun cheval fourni"}), 400

    df_nc = pd.DataFrame(chevaux)
    df_nc, _, _ = _enrichir(df_nc, note_mean, note_std)

    # Modèle principal
    df_nc['proba_principal'] = model.predict_proba(df_nc[FEATURES])[:, 1]

    # Modèle absolu
    df_nc['proba_absolu'] = modele_abs.predict_proba(df_nc[FEATURES_ABSOLU])[:, 1]

    # Top 3 principal (cote > 10)
    candidats = df_nc[df_nc['rapport'] > 10].copy()
    candidats = candidats.sort_values(['proba_principal','rapport'], ascending=[False,False])
    top3_principal = candidats.head(3)['numero'].tolist()

    # Top 3 absolu
    tous = df_nc.sort_values(['proba_absolu','rapport'], ascending=[False,False])
    top3_absolu = tous.head(3)['numero'].tolist()

    # ── Top features par cheval : rang relatif dans le peloton ──
    # Pour chaque cheval, on calcule son rang sur 4 dimensions clés
    # et on l'exprime en percentile (100 = meilleur du peloton)
    n = len(tous)

    # Résultat complet trié par proba absolu
    tous_list = []
    for i, (_, row) in enumerate(tous.iterrows()):
        rang_note    = int((tous['note']              > row['note']).sum())              + 1
        rang_rapport = int((tous['rapport']           < row['rapport']).sum())            + 1
        rang_valeur  = int((tous['score_valeur']      > row['score_valeur']).sum())      + 1
        rang_ratio   = int((tous['ratio_note_rapport']> row['ratio_note_rapport']).sum())+ 1

        def pct(rang): return round((1 - (rang - 1) / n) * 100)

        top_features = sorted([
            {"feature": "Note",          "valeur": float(row['note']),                       "score": pct(rang_note),    "rang": rang_note,    "total": n},
            {"feature": "Favori (cote)", "valeur": float(row['rapport']),                    "score": pct(rang_rapport), "rang": rang_rapport, "total": n},
            {"feature": "Valeur",        "valeur": round(float(row['score_valeur']), 2),     "score": pct(rang_valeur),  "rang": rang_valeur,  "total": n},
            {"feature": "Ratio note/cote","valeur": round(float(row['ratio_note_rapport']),3),"score": pct(rang_ratio),   "rang": rang_ratio,   "total": n},
        ], key=lambda x: x['score'], reverse=True)

        tous_list.append({
            "numero":          int(row['numero']),
            "note":            float(row['note']),
            "rapport":         float(row['rapport']),
            "proba_principal": round(float(row['proba_principal']) * 100, 1),
            "proba_absolu":    round(float(row['proba_absolu']) * 100, 1),
            "top3_principal":  int(row['numero']) in top3_principal,
            "top3_absolu":     int(row['numero']) in top3_absolu,
            "top_features":    top_features,
        })
    return jsonify({
        "tous": tous_list,
        "top3_principal": top3_principal,
        "top3_absolu":    top3_absolu,
    })


@app.route('/ajouter', methods=['POST'])
def ajouter():
    """
    Body JSON : {
      "date": "2026-03-01",
      "chevaux": [{"numero":1,"note":15,"rapport":12.5,"rang_arrivee":2}, ...]
    }

    Le réentraînement des modèles est lancé en tâche de fond (thread) pour
    ne pas bloquer la réponse HTTP — sur 117k lignes, l'entraînement prend
    plusieurs secondes et bloquerait le client sinon.
    """
    global df

    data    = request.get_json()
    date    = data.get('date')
    chevaux = data.get('chevaux', [])

    if not date or not chevaux:
        return jsonify({"error": "date et chevaux requis"}), 400

    # ── 1. Construire les nouvelles lignes ──
    rows = []
    for c in chevaux:
        rows.append({
            "date":         pd.to_datetime(date),
            "numero":       c['numero'],
            "note":         c['note'],
            "rapport":      c['rapport'],
            "rang_arrivee": c['rang_arrivee'],
            "score_cible":  0,
        })
    df_new = pd.DataFrame(rows)

    # ── 2. Persister immédiatement (PG ou CSV) ──
    _ecrire_courses_manuelles(df_new)

    # ── 3. Mettre à jour le DataFrame en mémoire ──
    df = pd.concat([df, df_new], ignore_index=True)

    # ── 4. Réentraînement asynchrone ──
    # On répond tout de suite au client, le réentraînement tourne en arrière-plan.
    # Les modèles restent utilisables avec les anciennes valeurs pendant ce temps.
    def _reentrainer_bg():
        global model, note_mean, note_std, modele_abs, note_mean_a, note_std_a
        try:
            print(f"🔄 Réentraînement en arrière-plan sur {len(df)} lignes…")
            df_p = df.copy()
            df_p['target'] = ((df_p['rapport'] >= 10) & (df_p['rang_arrivee'] <= 3)).astype(int)
            df_p, nm, ns = _enrichir(df_p)
            m = _entrainer(df_p, FEATURES, 'target')

            df_a = df.copy()
            df_a['target_absolu'] = (df_a['rang_arrivee'] <= 3).astype(int)
            df_a, nma, nsa = _enrichir(df_a)
            ma = _entrainer(df_a, FEATURES_ABSOLU, 'target_absolu')

            # Mise à jour atomique des globals
            model, note_mean, note_std = m, nm, ns
            modele_abs, note_mean_a, note_std_a = ma, nma, nsa
            print(f"✅ Réentraînement terminé ({len(df)} lignes / {df['date'].nunique()} courses)")
        except Exception as e:
            print(f"❌ Erreur réentraînement : {e}")

    threading.Thread(target=_reentrainer_bg, daemon=True).start()

    return jsonify({
        "message":   f"Course du {date} ajoutée — réentraînement en cours",
        "nb_lignes":  len(df),
        "nb_courses": int(df['date'].nunique()),
        "stockage":   "postgresql" if _get_pg() else "csv_ephemere",
    })


# ============================================================
# MODELE PMU — Chargement
# ============================================================
# MODELE PMU — Globals
# ============================================================
_model_pmu           = None
_features_pmu        = None
_le_driver           = None
_le_entr             = None
_driver_stats        = None
_entr_stats          = None
_duo_stats           = None
_spec_dist           = None
_spec_disc           = None
_prior_pmu           = None
_k_bayes_pmu         = None
_target_mean_pmu     = None
_target_std_pmu      = None
_ferrage_map_pmu     = None
_avis_map_pmu        = {'POSITIF': 1, 'NEUTRE': 0, 'NEGATIF': -1}
_mediane_rapport_ref = 18.0
_hist_snapshot       = None
_seuils_notes        = None

PMU_MODEL_PATH      = "model_pmu_v5.pkl"
PMU_V7_PATH         = "model_pmu_v15_attele.pkl"   # XGBoost trot attelé V15 ranking

# Modèles galop
GALOP_MODEL_PATHS = {
    'PLAT':  "model_pmu_plat_v4.pkl",    # XGBoost Ranking PLAT V4
    'HAIE':  "model_pmu_haie_v1.pkl",   # XGBoost Ranking HAIE V1
    'MONTE': "model_pmu_monte_v1.pkl",   # XGBoost Ranking MONTE V1
}
DISCIPLINES_TROT  = ('ATTELE', 'TROT_ATTELE')
DISCIPLINES_GALOP = ('PLAT', 'HAIE', 'MONTE')
DISCIPLINES_SKIP  = ('STEEPLECHASE', 'CROSS')

# Globals V9 (trot attelé)
_model_v7              = None
_bundle_v7             = {}
_use_v7                = False
_calibrator_v9         = None   # IsotonicRegression — calibre les probas
# Snapshots V9 — forme récente et momentum duo
_duo_momentum_snap     = None
_top3_3courses_snap    = None
_top3_60j_snap         = None
# Snapshots V10 — nouvelles features
_fraicheur_snap        = None   # fraicheur_jours par cheval
_progression_snap      = None   # progression_norm par cheval
_aptitude_snap         = None   # aptitude_piste par cheval/tranche
# Snapshots V12 — nouvelles features ranking
_niveau_snap           = None   # niveau_habituel par cheval
_confiance_seuils      = {'faible': 3.841, 'moyen': 5.028, 'fort': 5.398}
_fallback_rk_v9        = {'court': 76000, 'moyen': 75100,
                           'long': 76000, 'tres_long': 76500}
_duo_fiable_seuil_v9   = 5

# Globals galop
_models_galop       = {}   # {'PLAT': bundle, 'HAIE': bundle, 'MONTE': bundle}
_jockey_stats_galop = None # stats jockey pour PLAT et HAIE

# Globals PLAT V1 Ranking — snapshots spécifiques
_plat_jockey_stats        = None
_plat_duo_stats           = None
_plat_entr_stats          = None
_plat_top3_3c_snap        = None
_plat_top3_60j_snap       = None
_plat_aptitude_terrain    = None
_plat_aptitude_distance   = None
_plat_apt_dist_snap       = None
_plat_regularite_snap     = None
_plat_niveau_lot_snap     = None
_plat_niveau_snap         = None
_plat_jockey_hippo_stats  = None
_plat_aptitude_hippo_snap = None
_plat_confiance_seuils    = {'faible': 0.256, 'moyen': 0.325, 'fort': 0.403}
_plat_dernier_jockey_snap = None
_plat_apt_type_piste_snap = None
_plat_apt_terrain_label_snap = None

# Globals MONTE V1 Ranking
_monte_jockey_stats     = None
_monte_duo_stats        = None
_monte_entr_stats       = None
_monte_top3_3c_snap     = None
_monte_top3_60j_snap    = None
_monte_regularite_snap  = None
_monte_apt_dist_snap    = None
_monte_niveau_lot_snap  = None
_monte_niveau_snap      = None
_monte_confiance_seuils = {'faible': 0.096, 'moyen': 0.122, 'fort': 0.144}

# Globals HAIE V1 Ranking
_haie_jockey_stats    = None
_haie_entr_stats      = None
_haie_top3_3c_snap    = None
_haie_top3_60j_snap   = None
_haie_regularite_snap = None
_haie_apt_dist_snap   = None
_haie_niveau_lot_snap = None
_haie_niveau_snap     = None
_haie_confiance_seuils = {'faible': 0.083, 'moyen': 0.106, 'fort': 0.134}

DISC_MUSIQUE_MAP = {'a': 0, 'm': 1, 'p': 2, 'h': 3, 's': 4, 'c': 5}
DISCIPLINE_MAP   = {'TROT_ATTELE': 0, 'TROT_MONTE': 1, 'PLAT': 2, 'OBSTACLE': 3}
CORDE_MAP        = {'CORDE_A_GAUCHE': 0, 'CORDE_A_DROITE': 1}
SEXE_MAP         = {'MALES': 0, 'FEMELLES': 1, 'MIXTE': 2}


def _calculer_hist_snapshot():
    """Calcule les stats historiques par cheval depuis historique_notes.csv."""
    global _hist_snapshot
    hist = pd.read_csv(HISTORIQUE_PATH, usecols=['date','nom','rang_arrivee','rapport'])
    hist['date'] = pd.to_datetime(hist['date'])
    hist = hist.dropna(subset=['rang_arrivee']).sort_values(['nom','date'])

    # Calcul vectorisé
    g = hist.groupby('nom')
    nb       = g['rang_arrivee'].count()
    taux_top3= (hist['rang_arrivee'] <= 3).groupby(hist['nom']).mean()
    moy_cl   = g['rang_arrivee'].mean().round(2)
    moy_cote = g['rapport'].mean().round(2)

    # Tendance : moy 3 dernières - moy 3 premières (vectorisé)
    def _tendance(x):
        n = len(x)
        rec = x.iloc[-3:].mean()
        anc = x.iloc[:3].mean() if n >= 6 else x.mean()
        return round(float(anc - rec), 2)

    tendance = g['rang_arrivee'].apply(_tendance)

    # Courses dans les 60 derniers jours
    date_max  = hist['date'].max()
    date_60j  = date_max - pd.Timedelta(days=60)
    hist_rec  = hist[hist['date'] >= date_60j]
    courses_60j = hist_rec.groupby('nom')['rang_arrivee'].count().rename('courses_60j')

    _hist_snapshot = pd.DataFrame({
        'nom':                nb.index,
        'hist_nb':            nb.values,
        'hist_taux_top3':     taux_top3.values.round(3),
        'hist_moy_classement':moy_cl.values,
        'hist_tendance':      tendance.values,
        'hist_moy_cote':      moy_cote.values,
    }).join(courses_60j, on='nom').fillna({'courses_60j': 0})
    _hist_snapshot['courses_60j'] = _hist_snapshot['courses_60j'].astype(int)
    print(f"✅ hist_snapshot calculé : {len(_hist_snapshot)} chevaux")


def _charger_modele_pmu():
    global _model_pmu, _features_pmu, _le_driver, _le_entr
    global _driver_stats, _entr_stats, _duo_stats, _spec_dist, _spec_disc
    global _prior_pmu, _k_bayes_pmu
    global _target_mean_pmu, _target_std_pmu, _ferrage_map_pmu, _mediane_rapport_ref
    global _hist_snapshot, _seuils_notes

    if not os.path.exists(PMU_MODEL_PATH):
        print("⚠️  model_pmu.pkl introuvable — endpoint /notes_pmu désactivé")
        return False
    try:
        with open(PMU_MODEL_PATH, 'rb') as f:
            pmu = pickle.load(f)
        _model_pmu           = pmu['model']
        _features_pmu        = pmu['features']
        _le_driver           = pmu.get('le_driver')
        _le_entr             = pmu.get('le_entr')
        _driver_stats        = pmu.get('driver_stats')
        _entr_stats          = pmu.get('entr_stats')
        _duo_stats           = pmu.get('duo_stats')
        _spec_dist           = pmu.get('spec_dist')
        _spec_disc           = pmu.get('spec_disc')
        _prior_pmu           = pmu['prior']
        _k_bayes_pmu         = pmu['k_bayes']
        _target_mean_pmu     = pmu.get('target_mean')
        _target_std_pmu      = pmu.get('target_std')
        _ferrage_map_pmu     = pmu['ferrage_map']
        _mediane_rapport_ref = pmu.get('mediane_rapport_ref', 18.0)
        _hist_snapshot       = pmu.get('hist_snapshot')
        _seuils_notes        = pmu.get('seuils_notes')
        v = pmu.get('version', 1)
        nb_duos = len(_duo_stats) if _duo_stats is not None else 0
        print(f"✅ Modèle PMU v{v} chargé ({len(_features_pmu)} features, {nb_duos} duos)")

        # ── Calcul hist_snapshot depuis historique si absent du pkl ──
        if _hist_snapshot is None and os.path.exists(HISTORIQUE_PATH):
            try:
                _calculer_hist_snapshot()
            except Exception as e:
                print(f"⚠️  hist_snapshot non calculé : {e}")

        return True
    except Exception as e:
        print(f"❌ Erreur chargement model_pmu.pkl : {e}")
        return False


# ── Parseur musique v3 (format réel : position + discipline) ─
def _parser_musique_api(musique):
    from collections import Counter
    if not musique:
        return {
            'mus_nb_courses': 0, 'mus_nb_victoires': 0, 'mus_nb_podiums': 0,
            'mus_moy_classement': 99, 'mus_derniere_place': 99, 'mus_regularite': 0,
            'mus_nb_disq': 0, 'mus_taux_disq': 0.0,
            'mus_nb_tombes': 0, 'mus_nb_arretes': 0,
            'mus_tendance': 0.0, 'mus_score_pondere': 0.0,
            'mus_disc_principale': -1, 'mus_nb_disciplines': 0,
            'flag_disq_recente': 0.0,
        }
    clean   = re.sub(r'\(\d+\)', '', musique).strip()
    tokens  = re.findall(r'[0-9DATRdat][amphsc]', clean)
    if not tokens:
        return {
            'mus_nb_courses': 0, 'mus_nb_victoires': 0, 'mus_nb_podiums': 0,
            'mus_moy_classement': 99, 'mus_derniere_place': 99, 'mus_regularite': 0,
            'mus_nb_disq': 0, 'mus_taux_disq': 0.0,
            'mus_nb_tombes': 0, 'mus_nb_arretes': 0,
            'mus_tendance': 0.0, 'mus_score_pondere': 0.0,
            'mus_disc_principale': -1, 'mus_nb_disciplines': 0,
            'flag_disq_recente': 0.0,
        }
    entries, nb_disq, nb_tombes, nb_arretes = [], 0, 0, 0
    for tok in tokens[:10]:
        pos, disc = tok[0], tok[1].lower()
        if pos.isdigit():
            place = 10 if pos == '0' else int(pos)
        elif pos.upper() == 'D':
            place = 15; nb_disq += 1
        elif pos.upper() == 'T':
            place = 15; nb_tombes += 1
        elif pos.upper() == 'A':
            place = 15; nb_arretes += 1
        elif pos.upper() == 'R':
            place = 12
        else:
            continue
        entries.append((place, disc))
    if not entries:
        return {
            'mus_nb_courses': 0, 'mus_nb_victoires': 0, 'mus_nb_podiums': 0,
            'mus_moy_classement': 99, 'mus_derniere_place': 99, 'mus_regularite': 0,
            'mus_nb_disq': 0, 'mus_taux_disq': 0.0,
            'mus_nb_tombes': 0, 'mus_nb_arretes': 0,
            'mus_tendance': 0.0, 'mus_score_pondere': 0.0,
            'mus_disc_principale': -1, 'mus_nb_disciplines': 0,
            'flag_disq_recente': 0.0,
        }
    places      = [e[0] for e in entries]
    disciplines = [e[1] for e in entries]
    nb          = len(places)
    recentes    = places[:3]
    anciennes   = places[-3:] if nb >= 6 else places
    tendance    = round(float(np.mean(anciennes) - np.mean(recentes)), 2)
    poids       = [1.0 / (i + 1) for i in range(nb)]
    score_p     = round(sum(p*(10-min(pl,10)) for p,pl in zip(poids,places))/sum(poids), 3)
    disc_counter     = Counter(disciplines)
    disc_principale  = DISC_MUSIQUE_MAP.get(disc_counter.most_common(1)[0][0], -1)
    return {
        'mus_nb_courses':      nb,
        'mus_nb_victoires':    sum(1 for p in places if p == 1),
        'mus_nb_podiums':      sum(1 for p in places if p <= 3),
        'mus_moy_classement':  round(sum(places) / nb, 2),
        'mus_derniere_place':  places[0],
        'mus_regularite':      round(sum(1 for p in places if p <= 5) / nb, 2),
        'mus_nb_disq':         nb_disq,
        'mus_taux_disq':       round(nb_disq / nb, 2),
        'mus_nb_tombes':       nb_tombes,
        'mus_nb_arretes':      nb_arretes,
        'mus_tendance':        tendance,
        'mus_score_pondere':   score_p,
        'mus_disc_principale': disc_principale,
        'mus_nb_disciplines':  len(disc_counter),
        'flag_disq_recente':  round(min(places[0]/15, 1)*0.6 + (nb_disq/max(nb,1) > 0.3)*0.4, 3),
    }


def _perf_vide():
    return {
        'perf_nb': 0, 'perf_moy_classement': 99, 'perf_derniere_place': 99,
        'perf_nb_top3': 0, 'perf_taux_top3': 0.0,
        'perf_moy_rk': 0.0, 'perf_moy_gains': 0.0, 'perf_regularite': 0.0,
    }


def _fetch_performances(date_str, r_num, c_num):
    url = (f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
           f"/{date_str}/R{r_num}/C{c_num}/performances-detaillees")
    try:
        resp = http_requests.get(url, timeout=5)
        if resp.status_code in (400, 404, 204):
            return {}
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}
    result = {}
    for cheval in data.get('performancesDetaillees', []):
        num_pmu = cheval.get('numPmu')
        perfs   = cheval.get('performances', [])[:5]
        if not perfs:
            result[num_pmu] = _perf_vide(); continue
        classements, temps_list, gains_list = [], [], []
        for perf in perfs:
            cl = perf.get('ordreArrivee') or perf.get('classement')
            if cl and cl <= 15:
                classements.append(cl)
            t = perf.get('tempsObtenu') or perf.get('reductionKilometrique')
            if t and t > 0: temps_list.append(t)
            g = perf.get('gainsCourse') or perf.get('gains') or 0
            if g: gains_list.append(g)
        nb = len(classements)
        result[num_pmu] = {
            'perf_nb':             nb,
            'perf_moy_classement': round(sum(classements)/nb, 2) if nb > 0 else 99,
            'perf_derniere_place': classements[0] if classements else 99,
            'perf_nb_top3':        sum(1 for c in classements if c <= 3),
            'perf_taux_top3':      round(sum(1 for c in classements if c<=3)/nb,2) if nb>0 else 0.0,
            'perf_moy_rk':         round(sum(temps_list)/len(temps_list),1) if temps_list else 0.0,
            'perf_moy_gains':      round(sum(gains_list)/len(gains_list),1) if gains_list else 0.0,
            'perf_regularite':     round(sum(1 for c in classements if c<=5)/nb,2) if nb>0 else 0.0,
        }
    return result


def _fetch_conditions(date_str, r_num, c_num):
    url = f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme/{date_str}"
    try:
        resp = http_requests.get(url, timeout=5)
        if resp.status_code in (400, 404, 204):
            return _cond_vides()
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return _cond_vides()
    for reunion in data.get('programme', {}).get('reunions', []):
        if reunion.get('numOfficiel') == r_num or reunion.get('numReunion') == r_num:
            for course in reunion.get('courses', []):
                if course.get('numOrdre') == c_num or course.get('numExterne') == c_num:
                    penet = course.get('penetrometre', {}) or {}
                    try:
                        terrain_val = float(str(penet.get('valeurMesure','3')).replace(',','.'))
                    except:
                        terrain_val = 3.0
                    terrain_label = penet.get('intitule', '') or ''
                    hippo = reunion.get('hippodrome', {}) or {}
                    return {
                        'distance':        course.get('distance', 0) or 0,
                        'montant_prix':    course.get('montantPrix', 0) or 0,
                        'discipline':      DISCIPLINE_MAP.get(course.get('discipline',''), 0),
                        'discipline_raw':  course.get('discipline', ''),
                        'corde':           CORDE_MAP.get(course.get('corde',''), 0),
                        'condition_sexe':  SEXE_MAP.get(course.get('conditionSexe',''), 2),
                        'nb_partants':     course.get('nombreDeclaresPartants', 0) or 0,
                        'terrain_val':     terrain_val,
                        'terrain_label':   terrain_label,
                        'hippodrome_code': hippo.get('codeHippodrome', '') or '',
                        'hippodrome_nom':  hippo.get('libelleCourt', '') or '',
                    }
    return _cond_vides()


def _cond_vides():
    return {'distance': 0, 'montant_prix': 0, 'discipline': 0, 'discipline_raw': '',
            'corde': 0, 'condition_sexe': 2, 'nb_partants': 0,
            'terrain_val': 3.0, 'terrain_label': '',
            'hippodrome_code': '', 'hippodrome_nom': ''}


# Seuils par défaut si bundle ne les contient pas (utilisé pour V6 fallback)
_SEUILS_DEFAUT = [
    (0.05, 1), (0.10, 2), (0.15, 3), (0.20, 4), (0.25, 5),
    (0.30, 6), (0.35, 7), (0.40, 8), (0.45, 9), (0.50, 10),
    (0.55, 11), (0.60, 12), (0.65, 13), (0.70, 14), (0.75, 15),
    (0.80, 16), (0.85, 17), (0.90, 18), (0.95, 19), (1.01, 20),
]

def _proba_to_note_api(proba_series):
    """
    Conversion V6 (fallback) : seuils fixes sur probabilités absolues.
    """
    seuils = _seuils_notes if _seuils_notes is not None else _SEUILS_DEFAUT
    def _convert(p):
        for seuil, note in seuils:
            if p < seuil:
                return note
        return 20
    return pd.Series(proba_series).apply(_convert)


def _proba_to_note_v7(scores_series, proba_min_ref=None, proba_max_ref=None):
    """
    Conversion score/proba → note sur 20.

    Fonctionne pour les deux types de modèles :

    ── Ranking (V12+) ──────────────────────────────────────
    Les scores XGBRanker sont déjà des scores de ranking
    calibrés sur le peloton. On normalise linéairement sur
    la plage [min, max] du peloton courant.
    → Les écarts proportionnels entre chevaux sont préservés.
    → Le #1 a toujours 20, le dernier toujours 1 — VOULU
      car le modèle a appris à ordonner les chevaux.
    → Si cheval A=0.52 et B=0.51 (proches) → notes proches.
    → Si A=0.52 et B=-0.30 (dominant) → grand écart de notes.

    ── Classification (V9-V11) ─────────────────────────────
    Les probas sont des valeurs absolues [0,1].
    On normalise sur les bornes proba_min/proba_max du pkl
    pour éviter la saturation à 20 quand les probas sont
    toutes élevées.
    """
    s         = pd.Series(scores_series).reset_index(drop=True)
    orig_idx  = scores_series.index if hasattr(scores_series, 'index') else s.index

    if len(s) == 1:
        notes = pd.Series([10], index=s.index)
        notes.index = orig_idx
        return notes

    # ── Détection du type de modèle ──────────────────────
    is_ranking = _bundle_v7.get('model_type', 'classification') == 'ranking'

    if is_ranking:
        # ── Ranking : normalisation sur le peloton courant ──
        # Linéaire stricte — préserve les vrais écarts
        s_min, s_max = s.min(), s.max()
        plage = s_max - s_min

        if plage < 1e-6:
            notes = pd.Series([10] * len(s), index=s.index)
            notes.index = orig_idx
            return notes

        notes_float = (s - s_min) / plage * 19 + 1

    else:
        # ── Classification : normalisation sur bornes du pkl ──
        # Utilise proba_min/proba_max pour éviter la saturation
        p_min = proba_min_ref if proba_min_ref is not None \
                else _bundle_v7.get('proba_min', s.min())
        p_max = proba_max_ref if proba_max_ref is not None \
                else _bundle_v7.get('proba_max', s.max())
        plage = p_max - p_min

        if plage < 1e-6:
            notes = pd.Series([10] * len(s), index=s.index)
            notes.index = orig_idx
            return notes

        notes_float = (s - p_min) / plage * 19 + 1

    notes = notes_float.round().clip(1, 20).astype(int)
    notes.index = orig_idx
    return notes







def _norm_g(s, lo, hi):
    return ((s.clip(lo, hi) - lo) / (hi - lo + 1e-9)).clip(0, 1)

def _norm_rel_g(s):
    mn, mx = s.min(), s.max()
    if mx - mn < 1e-9: return pd.Series(0.5, index=s.index)
    return ((s - mn) / (mx - mn)).clip(0, 1)

def _norm_mix_g(s, lo, hi, rel=0.5):
    return (_norm_g(s, lo, hi) * (1-rel) + _norm_rel_g(s) * rel).clip(0, 1)

def _notes_pmu_plat_v1(df_nc, date_str, r_num, c_num):
    """Pipeline inférence PLAT V1 — XGBoost Ranking."""
    bundle  = _models_galop['PLAT']
    model   = bundle['model']
    feats   = bundle['features']
    cal     = bundle.get('calibrator')
    prior   = bundle.get('prior_win', 0.294)
    k_bayes = bundle.get('k_bayes', 10)
    fallback = prior * k_bayes / (k_bayes + 1)
    n       = len(df_nc)

    # ── Features simples ─────────────────────────────────────
    df_nc['ratio_victoires']  = df_nc['nb_victoires']  / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places']     = df_nc['nb_places']      / (df_nc['nb_courses'] + 1)
    df_nc['gains_par_course'] = df_nc['gains_carriere'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_gains_rec']  = df_nc['gains_annee']    / (df_nc['gains_carriere'] + 1)
    df_nc['log_montant_prix'] = np.log1p(df_nc['montant_prix'])
    df_nc['nb_partants_c']    = n
    df_nc['est_3ans']         = (df_nc['age'] == 3).astype(float)
    # V11 : ratio_podiums remplace mus_nb_podiums brute (corr +0.090 -> +0.164)
    df_nc['ratio_podiums']    = df_nc['mus_nb_podiums'] / (df_nc['mus_nb_courses'] + 1)

    # V15 : changement_jockey supprime du modele (ignore par LightGBM)

    # Tranche distance PLAT
    df_nc['tranche_distance'] = pd.cut(df_nc['distance'],
        bins=[0, 1200, 1600, 2000, 9999],
        labels=['sprint','mile','intermediaire','long']).astype(str)

    # V19 Option A : 3 categories terrain (robustesse)
    tv = pd.to_numeric(df_nc['terrain_val'], errors='coerce').fillna(3.4)
    df_nc['terrain_cat'] = pd.cut(tv,
        bins=[0, 3.2, 3.8, 99],
        labels=['rapide','souple','lourd']).astype(str)

    # V18 : type_piste (HERBE/GAZON/PSF/DIRT/SABLE/INCONNU)
    if 'type_piste' not in df_nc.columns:
        df_nc['type_piste'] = 'INCONNU'
    df_nc['type_piste'] = df_nc['type_piste'].fillna('INCONNU').astype(str)
    df_nc.loc[df_nc['type_piste'].isin(['nan','None','']), 'type_piste'] = 'INCONNU'

    # ── Features relatives au peloton ────────────────────────
    # Handicap valeur — V11 : ecart_handicap_peloton supprimé (redondant 96% avec rang_handicap_norm)
    hv = df_nc['handicap_valeur'].values.astype(float)
    hv_valides = hv[hv > 0]
    if len(hv_valides) < 2:
        df_nc['rang_handicap_norm']     = 0.5
        df_nc['ecart_handicap_peloton'] = 0.5  # conservé pour compat ancien modèle
    else:
        hv_for_rank = np.where(hv > 0, hv, np.nan)
        hv_rank = pd.Series(hv_for_rank).rank(ascending=False, na_option='keep').values
        n_valides = (~np.isnan(hv_for_rank)).sum()
        df_nc['rang_handicap_norm'] = np.where(
            np.isnan(hv_rank), 0.5,
            1 - (hv_rank - 1) / max(n_valides - 1, 1))
        # ecart_handicap_peloton conservé pour compat si ancien modèle
        hv_mean = hv_valides.mean()
        hv_std  = hv_valides.std() + 1e-9
        df_nc['ecart_handicap_peloton'] = np.where(
            hv > 0, ((hv - hv_mean) / hv_std).clip(-3, 3) / 3 * 0.5 + 0.5, 0.5)

    # Poids relatif
    pw      = df_nc['handicap_poids'].values.astype(float)
    pw_std  = pw.std()
    df_nc['poids_relatif_peloton'] = (
        -(pw - pw.mean()) / (pw_std + 1e-9)).clip(-3, 3) / 3 * 0.5 + 0.5

    # Stalle normalisée
    sc = df_nc['place_corde'].values.astype(float)
    sc_max = sc.max()
    df_nc['stalle_norm'] = (1 - (sc - 1) / (sc_max - 1)) if sc_max > 1 else 0.5

    # Rang cote peloton
    cote     = df_nc['rapport_ref'].values.astype(float)
    cote_rank = pd.Series(cote).rank(ascending=True).values
    df_nc['rang_cote_peloton'] = 1 - (cote_rank - 1) / max(n - 1, 1)

    # Écart cotes normalisé
    ec     = df_nc['ecart_cotes'].values.astype(float)
    ec_min, ec_max = ec.min(), ec.max()
    df_nc['ecart_cotes_norm'] = (
        (-ec - (-ec).min()) / ((-ec).max() - (-ec).min() + 1e-9)).clip(0, 1)

    # Gains relatifs — V8 : gains_par_course au lieu de gains_carriere
    # Resout biais d'age (vieux chevaux avaient ratio eleve mais top3 faible)
    gpc = (df_nc['gains_carriere'].values.astype(float) /
           (df_nc['nb_courses'].values.astype(float) + 1))
    gpc_moy = gpc.mean()
    df_nc['ratio_gains_peloton'] = (
        (gpc / (gpc_moy + 1e-9)).clip(0, 5) / 5 if gpc_moy > 0 else 0.5)

    # ── Jockey stats — V9 : jockey_win_rate_30j supprime (redondant avec bayes) ──
    if _plat_jockey_stats is not None:
        try:
            cols_jky = _plat_jockey_stats.columns
            cols_needed = ['driver', 'jockey_win_rate_bayes']
            if 'jockey_n' in cols_jky: cols_needed.append('jockey_n')
            if 'jockey_win_rate_30j' in cols_jky: cols_needed.append('jockey_win_rate_30j')
            df_nc = df_nc.merge(_plat_jockey_stats[cols_needed], on='driver', how='left')
        except Exception as e:
            print(f"⚠️  PLAT jockey_stats merge échoué ({e})")
    for col, val in [('jockey_win_rate_bayes', fallback),
                     ('jockey_win_rate_30j',   fallback),
                     ('jockey_n',              0)]:
        if col not in df_nc.columns: df_nc[col] = val
        df_nc[col] = df_nc[col].fillna(val)

    # Rang jockey dans le peloton
    jwr   = df_nc['jockey_win_rate_bayes'].values.astype(float)
    j_rank = pd.Series(jwr).rank(ascending=False).values
    df_nc['rang_jockey_peloton'] = 1 - (j_rank - 1) / max(n - 1, 1)

    # V14 : rang gains_par_course dans le peloton (corr +0.184 vs +0.040 pour brut)
    gpc = df_nc['gains_par_course'].values.astype(float)
    g_rank = pd.Series(gpc).rank(ascending=False).values
    df_nc['rang_gains_peloton'] = 1 - (g_rank - 1) / max(n - 1, 1)

    # Duo jockey × cheval
    if _plat_duo_stats is not None:
        try:
            df_nc = df_nc.merge(
                _plat_duo_stats[['nom','driver','duo_jockey_win_rate']],
                on=['nom','driver'], how='left')
        except Exception:
            pass
    if 'duo_jockey_win_rate' not in df_nc.columns:
        df_nc['duo_jockey_win_rate'] = fallback
    df_nc['duo_jockey_win_rate'] = df_nc['duo_jockey_win_rate'].fillna(fallback)

    # ── Entraîneur stats ─────────────────────────────────────
    # V17 : ajout entr_forme_60j (forme récente de l'écurie)
    if _plat_entr_stats is not None:
        try:
            cols_entr = ['entraineur', 'entr_win_rate_bayes']
            if 'entr_win_rate_30j' in _plat_entr_stats.columns:
                cols_entr.append('entr_win_rate_30j')
            if 'entr_forme_60j' in _plat_entr_stats.columns:
                cols_entr.append('entr_forme_60j')
            df_nc = df_nc.merge(_plat_entr_stats[cols_entr], on='entraineur', how='left')
        except Exception as e:
            print(f"⚠️  PLAT entr merge échoué ({e})")
    for col, val in [('entr_win_rate_bayes', fallback),
                     ('entr_win_rate_30j',   fallback),
                     ('entr_forme_60j',      fallback)]:
        if col not in df_nc.columns: df_nc[col] = val
        df_nc[col] = df_nc[col].fillna(val)

    # V17 : rang_entraineur_peloton (rang de entr_win_rate_bayes dans peloton)
    # Résiduelle +0.059 au-delà de entr_win_rate_bayes
    ewr = df_nc['entr_win_rate_bayes'].values.astype(float)
    e_rank = pd.Series(ewr).rank(ascending=False).values
    df_nc['rang_entraineur_peloton'] = 1 - (e_rank - 1) / max(n - 1, 1)

    # ── Forme récente ─────────────────────────────────────────
    if _plat_top3_3c_snap is not None:
        try:
            snap = _plat_top3_3c_snap.copy().reset_index(drop=True)
            if 'top3_3courses' in snap.columns and 'nom' in snap.columns:
                df_nc = df_nc.merge(snap[['nom','top3_3courses']], on='nom', how='left')
        except Exception:
            pass
    if 'top3_3courses' not in df_nc.columns: df_nc['top3_3courses'] = prior
    df_nc['top3_3courses'] = df_nc['top3_3courses'].fillna(prior)

    if _plat_top3_60j_snap is not None:
        try:
            snap = _plat_top3_60j_snap.copy().reset_index(drop=True)
            if 'top3_60j' in snap.columns and 'nom' in snap.columns:
                df_nc = df_nc.merge(snap[['nom','top3_60j']], on='nom', how='left')
        except Exception:
            pass
    if 'top3_60j' not in df_nc.columns: df_nc['top3_60j'] = prior
    df_nc['top3_60j'] = df_nc['top3_60j'].fillna(prior)

    # ── Aptitude distance RÉCENTE (V2) ───────────────────────
    if _plat_apt_dist_snap is not None:
        try:
            snap = _plat_apt_dist_snap.copy().reset_index(drop=True)
            if 'apt_dist_recente' in snap.columns:
                df_nc = df_nc.merge(
                    snap[['nom','tranche_distance','apt_dist_recente']],
                    on=['nom','tranche_distance'], how='left')
        except Exception:
            pass
    if 'apt_dist_recente' not in df_nc.columns:
        df_nc['apt_dist_recente'] = prior
    df_nc['apt_dist_recente'] = df_nc['apt_dist_recente'].fillna(prior)

    # ── Aptitude terrain — V14 : apt_terrain_actuel depuis snap (nom, terrain_cat) ──
    if _plat_aptitude_terrain is not None:
        try:
            snap = _plat_aptitude_terrain.copy().reset_index(drop=True)
            # V14 : snap contient ['nom','terrain_cat','apt_terrain_actuel']
            if 'apt_terrain_actuel' in snap.columns:
                df_nc = df_nc.merge(
                    snap[['nom','terrain_cat','apt_terrain_actuel']],
                    on=['nom','terrain_cat'], how='left')
            # Retrocompat : ancien snap avec 'aptitude_terrain'
            elif 'aptitude_terrain' in snap.columns:
                df_nc = df_nc.merge(
                    snap[['nom','terrain_cat','aptitude_terrain']],
                    on=['nom','terrain_cat'], how='left')
                df_nc['apt_terrain_actuel'] = df_nc['aptitude_terrain']
        except Exception as e:
            print(f"⚠️  PLAT apt_terrain merge échoué ({e})")
    if 'apt_terrain_actuel' not in df_nc.columns: df_nc['apt_terrain_actuel'] = prior
    df_nc['apt_terrain_actuel'] = df_nc['apt_terrain_actuel'].fillna(prior)
    # Compat : garder aussi aptitude_terrain pour ancien code
    if 'aptitude_terrain' not in df_nc.columns: df_nc['aptitude_terrain'] = prior
    df_nc['aptitude_terrain'] = df_nc['aptitude_terrain'].fillna(prior)

    # V18 : apt_type_piste (aptitude cheval x type_piste)
    if _plat_apt_type_piste_snap is not None:
        try:
            snap_tp = _plat_apt_type_piste_snap.copy().reset_index(drop=True)
            if 'apt_type_piste' in snap_tp.columns:
                df_nc = df_nc.merge(
                    snap_tp[['nom','type_piste','apt_type_piste']],
                    on=['nom','type_piste'], how='left')
        except Exception as e:
            print(f"⚠️  PLAT apt_type_piste merge échoué ({e})")
    if 'apt_type_piste' not in df_nc.columns: df_nc['apt_type_piste'] = prior
    df_nc['apt_type_piste'] = df_nc['apt_type_piste'].fillna(prior)

    # V19B : apt_terrain_label (aptitude cheval x terrain_label textuel)
    if 'terrain_label' not in df_nc.columns:
        df_nc['terrain_label'] = 'INCONNU'
    df_nc['terrain_label'] = df_nc['terrain_label'].fillna('INCONNU').astype(str)
    df_nc.loc[df_nc['terrain_label'].isin(['nan','None','']), 'terrain_label'] = 'INCONNU'
    if _plat_apt_terrain_label_snap is not None:
        try:
            snap_tl = _plat_apt_terrain_label_snap.copy().reset_index(drop=True)
            if 'apt_terrain_label' in snap_tl.columns:
                df_nc = df_nc.merge(
                    snap_tl[['nom','terrain_label','apt_terrain_label']],
                    on=['nom','terrain_label'], how='left')
        except Exception as e:
            print(f"⚠️  PLAT apt_terrain_label merge échoué ({e})")
    if 'apt_terrain_label' not in df_nc.columns: df_nc['apt_terrain_label'] = prior
    df_nc['apt_terrain_label'] = df_nc['apt_terrain_label'].fillna(prior)

    # ── Aptitude distance ────────────────────────────────────
    if _plat_aptitude_distance is not None:
        try:
            snap = _plat_aptitude_distance.copy().reset_index(drop=True)
            if 'aptitude_distance' in snap.columns:
                df_nc = df_nc.merge(
                    snap[['nom','tranche_distance','aptitude_distance']],
                    on=['nom','tranche_distance'], how='left')
        except Exception:
            pass
    if 'aptitude_distance' not in df_nc.columns: df_nc['aptitude_distance'] = prior
    df_nc['aptitude_distance'] = df_nc['aptitude_distance'].fillna(prior)

    # ── Aptitude hippodrome (V4) ─────────────────────────────
    # Récupérer le code hippodrome depuis les données de la course
    # Il est passé dans df_nc via le champ 'hippodrome_code' si disponible
    if 'hippodrome_code' not in df_nc.columns:
        df_nc['hippodrome_code'] = ''

    if _plat_aptitude_hippo_snap is not None and 'hippodrome_code' in df_nc.columns:
        try:
            snap = _plat_aptitude_hippo_snap[['nom','hippodrome_code','aptitude_hippodrome']]
            df_nc = df_nc.merge(snap, on=['nom','hippodrome_code'], how='left')
        except Exception:
            pass
    if 'aptitude_hippodrome' not in df_nc.columns:
        df_nc['aptitude_hippodrome'] = prior
    df_nc['aptitude_hippodrome'] = df_nc['aptitude_hippodrome'].fillna(prior)

    # ── Jockey win rate par hippodrome (V4) ──────────────────
    if _plat_jockey_hippo_stats is not None and 'hippodrome_code' in df_nc.columns:
        try:
            snap = _plat_jockey_hippo_stats[['driver','hippodrome_code','jockey_win_rate_hippo']]
            df_nc = df_nc.merge(snap, on=['driver','hippodrome_code'], how='left')
        except Exception:
            pass
    if 'jockey_win_rate_hippo' not in df_nc.columns:
        df_nc['jockey_win_rate_hippo'] = fallback
    df_nc['jockey_win_rate_hippo'] = df_nc['jockey_win_rate_hippo'].fillna(fallback)
    if _plat_niveau_snap is not None:
        try:
            df_nc = df_nc.merge(
                _plat_niveau_snap[['nom','niveau_habituel']], on='nom', how='left')
        except Exception:
            pass
    if 'niveau_habituel' not in df_nc.columns:
        df_nc['niveau_habituel'] = df_nc['montant_prix']
    df_nc['niveau_habituel']  = df_nc['niveau_habituel'].fillna(df_nc['montant_prix'])
    df_nc['ratio_niveau']     = (df_nc['montant_prix'] /
                                  (df_nc['niveau_habituel'] + 1)).clip(0, 5)
    df_nc['descente_niveau']  = (df_nc['ratio_niveau'] < 0.8).astype(float)

    # ── Régularité top3 (V2) ─────────────────────────────────
    if _plat_regularite_snap is not None:
        try:
            snap = _plat_regularite_snap.copy().reset_index(drop=True)
            if 'regularite_top3' in snap.columns and 'nom' in snap.columns:
                df_nc = df_nc.merge(snap[['nom','regularite_top3']],
                                    on='nom', how='left')
        except Exception:
            pass
    if 'regularite_top3' not in df_nc.columns:
        df_nc['regularite_top3'] = 0.0
    df_nc['regularite_top3'] = df_nc['regularite_top3'].fillna(0.0)

    # ── Niveau lot récent (V2) ────────────────────────────────
    if _plat_niveau_lot_snap is not None:
        try:
            snap = _plat_niveau_lot_snap.copy().reset_index(drop=True)
            if 'niveau_lot_recent' in snap.columns and 'nom' in snap.columns:
                df_nc = df_nc.merge(snap[['nom','niveau_lot_recent']],
                                    on='nom', how='left')
        except Exception:
            pass
    if 'niveau_lot_recent' not in df_nc.columns:
        df_nc['niveau_lot_recent'] = df_nc['montant_prix']
    df_nc['niveau_lot_recent'] = df_nc['niveau_lot_recent'].fillna(df_nc['montant_prix'])
    df_nc['ratio_niveau_lot']  = (df_nc['montant_prix'] /
                                   (df_nc['niveau_lot_recent'] + 1)).clip(0, 5)
    df_nc['descente_lot']      = (df_nc['ratio_niveau_lot'] < 0.8).astype(float)

    # ── Prédiction ranking ────────────────────────────────────
    df_input = pd.DataFrame(index=df_nc.index)
    for feat in feats:
        if feat in df_nc.columns:
            df_input[feat] = df_nc[feat]
        else:
            print(f"⚠️  PLAT feature '{feat}' absente → fallback {fallback:.3f}")
            df_input[feat] = fallback

    # Détection automatique XGBoost vs LightGBM Booster
    if hasattr(model, 'predict_proba'):
        scores_bruts = model.predict(df_input[feats])
    else:
        scores_bruts = np.asarray(model.predict(df_input[feats]))
    score_brut   = pd.Series(scores_bruts, index=df_nc.index)

    # Notes sur 20 depuis scores bruts (ranking linéaire sur le peloton courant)
    # Normalisation simple pour modèle ranking PLAT (XGBoost ou LightGBM)
    s_min, s_max = float(score_brut.min()), float(score_brut.max())
    plage_note = s_max - s_min
    if plage_note < 1e-6:
        df_nc['note_pmu'] = 10
    else:
        df_nc['note_pmu'] = ((score_brut - s_min) / plage_note * 19 + 1).round().clip(1, 20).astype(int)

    # Indice de confiance
    plage_scores = float(score_brut.max() - score_brut.min())
    seuils       = _plat_confiance_seuils
    if plage_scores < seuils.get('faible', 0.347):
        confiance_course = 'faible'
    elif plage_scores > seuils.get('fort', 0.845):
        confiance_course = 'fort'
    else:
        confiance_course = 'moyen'
    df_nc['_plage_scores'] = plage_scores
    df_nc['_confiance']    = confiance_course

    # Calibrateur → proba_pmu (affichage %)
    if cal is not None:
        try:
            score_final = pd.Series(
                cal.predict(score_brut.values), index=df_nc.index)
        except Exception:
            score_final = score_brut
    else:
        score_final = score_brut
    df_nc['proba_pmu'] = score_final

    # Score value
    if '_cote_app' in df_nc.columns:
        cote_val = df_nc['_cote_app'].fillna(10.0).clip(1.1, 200)
        df_nc['score_value'] = (df_nc['note_pmu'] * np.log(cote_val)).round(1)
    else:
        df_nc['score_value'] = 0.0

    # Scores métier — vraies dimensions V4 (pour affichage frontend)
    # Normalisées sur [0, 100] pour les barres de progression
    df_nc['score_forme']      = df_nc['mus_score_pondere'].fillna(0) / 9          # 0-9 → 0-100
    df_nc['score_jockey']     = df_nc['rang_jockey_peloton'].fillna(0.5)          # 0-1 → affiché *100
    df_nc['score_handicap']   = df_nc['rang_handicap_norm'].fillna(0.5)           # 0-1 → affiché *100
    df_nc['score_historique'] = df_nc['top3_3courses'].fillna(prior)              # 0-1 → affiché *100
    df_nc['score_gains']      = df_nc['ratio_victoires'].fillna(0).clip(0, 0.5) * 2  # 0-0.5 → 0-1
    df_nc['score_niveau']     = (1 - df_nc['ratio_niveau_lot'].fillna(1).clip(0, 2) / 2)  # descente=1
    df_nc['score_cote']       = 0.5  # non utilisé en V4 — cotes exclues

    # JSON
    result = []
    for _, row in df_nc.sort_values('note_pmu', ascending=False).iterrows():
        result.append({
            "numero":    int(row['numero']),
            "nom":       str(row['nom']),
            "note_pmu":  int(row['note_pmu']),
            "proba_pmu": round(float(row['proba_pmu']) * 100, 1) if pd.notna(row['proba_pmu']) else 0,
            "driver":    str(row['driver']),
            "cote":      float(row['_cote_app']) if pd.notna(row.get('_cote_app')) else None,
            "avis":      int(row['avis_entraineur']) if pd.notna(row.get('avis_entraineur')) else 0,
            "scores": {
                "forme":      int(round(float(row['score_forme'])     * 100)),
                "duo":        int(round(float(row['score_jockey'])    * 100)),
                "jockey":     int(round(float(row['score_jockey'])    * 100)),
                "historique": int(round(float(row['score_historique'])* 100)),
                "gains":      int(round(float(row['score_gains'])     * 100)),
                "handicap":   int(round(float(row['score_handicap'])  * 100)),
                "niveau":     int(round(float(row['score_niveau'])    * 100)),
                "cote":       int(round(float(row['score_cote'])      * 100)),
            },
            "taux_disq":       round(float(row['mus_taux_disq']) * 100, 1) if pd.notna(row.get('mus_taux_disq')) else 0,
            "musique":         str(row.get('musique', '')) if row.get('musique') else '',
            "handicap_poids":  int(row.get('handicap_poids', 0)),
            "handicap_valeur": float(row.get('handicap_valeur', 0)),
            "score_value":     round(float(row.get('score_value', 0)), 1),
            "nb_courses_base": int(row.get('mus_nb_courses', 0)),
            "rk_brut":         float(row['rk_brut']) if pd.notna(row.get('rk_brut')) and row.get('rk_brut') else None,
            "flag_chrono":     str(row.get('flag_chrono', 'ok')),
            "deferre":         str(row.get('cat_ferrure', 'FERRE')),
            "rk_ferrure":      float(row['reduction_km_v2_ferrure']) if pd.notna(row.get('reduction_km_v2_ferrure')) and row.get('reduction_km_v2_ferrure') else None,
            "tendance_chrono": str(row.get('tendance_chrono', 'inconnu')),
        })

    return jsonify({
        "date":      date_str,
        "reunion":   r_num,
        "course":    c_num,
        "discipline":"PLAT",
        "version":   bundle.get('version', 'plat_v4_ranking'),
        "chevaux":   result,
        "confiance": confiance_course,
        "plage":     round(plage_scores, 3),
    })


def _notes_pmu_galop(df_nc, discipline_raw, date_str, r_num, c_num):
    """Pipeline de scoring et prédiction pour les disciplines galop."""

    # ── PLAT V1 Ranking — pipeline dédié ─────────────────────
    # Accepte 'ranking', 'xgboost', 'lightgbm' (PLAT V8 sauve 'xgboost'/'lightgbm')
    bundle_plat = _models_galop.get('PLAT')
    if discipline_raw == 'PLAT' and bundle_plat and \
       bundle_plat.get('model_type') in ('ranking', 'xgboost', 'lightgbm'):
        return _notes_pmu_plat_v1(df_nc, date_str, r_num, c_num)

    # ── MONTE V1 Ranking — pipeline dédié ────────────────────
    bundle_monte = _models_galop.get('MONTE')
    if discipline_raw == 'MONTE' and bundle_monte and \
       bundle_monte.get('model_type') == 'ranking':
        return _notes_pmu_monte_v1(df_nc, date_str, r_num, c_num)

    # ── HAIE V1 Ranking — pipeline dédié ─────────────────────
    bundle_haie = _models_galop.get('HAIE')
    if discipline_raw == 'HAIE' and bundle_haie and \
       bundle_haie.get('model_type') == 'ranking':
        return _notes_pmu_haie_v1(df_nc, date_str, r_num, c_num)

    prior    = 0.15
    k_bayes  = 10
    fallback = prior * k_bayes / (k_bayes + 1)

    # ── Features dérivées ────────────────────────────────────
    df_nc['ratio_victoires']  = df_nc['nb_victoires'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places']     = df_nc['nb_places']    / (df_nc['nb_courses'] + 1)
    df_nc['gains_par_course'] = df_nc['gains_carriere'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_gains_rec']  = df_nc['gains_annee'] / (df_nc['gains_carriere'] + 1)
    df_nc['ratio_places_second']    = df_nc['nb_places_second']    / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places_troisieme'] = df_nc['nb_places_troisieme'] / (df_nc['nb_courses'] + 1)
    df_nc['rang_cote_course'] = df_nc['rapport_ref'].rank(ascending=True, method='min')
    nb_ch = len(df_nc)
    df_nc['rang_cote_norme']  = (df_nc['rang_cote_course'] - 1) / (nb_ch - 1 + 1e-8)
    df_nc['rang_poids']   = df_nc['handicap_poids'].rank(ascending=True, method='min')
    df_nc['rang_poids_n'] = (df_nc['rang_poids'] - 1) / (nb_ch - 1 + 1e-8)

    # ── Score forme ──────────────────────────────────────────
    s_score_p  = _norm_mix_g(df_nc['mus_score_pondere'], 0, 9)
    s_derniere = _norm_mix_g(15 - df_nc['mus_derniere_place'], 0, 14)
    s_podiums  = _norm_mix_g(df_nc['mus_nb_podiums'], 0, 5)
    s_disq     = 1 - _norm_g(df_nc['mus_taux_disq'], 0, 0.3)
    # Bug fix : pénaliser fortement les disqualifications récentes (3 dernières courses)
    def _nb_disq_recent_g(musique):
        import re as _re
        if not musique: return 0
        clean = _re.sub(r'\(\d+\)', '', str(musique)).strip()
        tokens = _re.findall(r'[0-9DATRdat][amphsc]', clean)[:3]
        return sum(1 for t in tokens if t[0].upper() == 'D')
    df_nc['_nb_disq_recent'] = df_nc['musique'].apply(_nb_disq_recent_g) if 'musique' in df_nc.columns else 0
    s_disq_recent = (2 - df_nc['_nb_disq_recent'].clip(0, 2)) / 2
    s_chutes   = 1 - _norm_g(df_nc['mus_nb_tombes'], 0, 3)
    s_age      = 1 - _norm_g(df_nc['age'].fillna(5), 3, 12)
    df_nc['score_forme'] = (
        s_score_p     * 0.25 +
        s_derniere    * 0.20 +
        s_podiums     * 0.12 +
        s_disq        * 0.08 +
        s_disq_recent * 0.20 +
        s_chutes      * 0.08 +
        s_age         * 0.07
    ).clip(0, 1)
    df_nc.drop(columns=['_nb_disq_recent'], inplace=True, errors='ignore')

    # ── Score duo / jockey selon discipline ────────────────────
    if discipline_raw in ('PLAT', 'HAIE'):
        # PLAT et HAIE : score jockey seul depuis historique_galop.csv
        df_nc['jockey_win_rate_bayes'] = fallback
        df_nc['jockey_n']              = 0
        jockey_source = _jockey_stats_galop if _jockey_stats_galop is not None else (
            _driver_stats if _driver_stats is not None and 'driver_win_rate_bayes' in _driver_stats.columns else None
        )
        if jockey_source is not None:
            # Détection automatique : snapshots récents utilisent jockey_*, anciens driver_*
            cols = jockey_source.columns
            wr_col = 'jockey_win_rate_bayes' if 'jockey_win_rate_bayes' in cols else 'driver_win_rate_bayes'
            n_col  = 'jockey_n' if 'jockey_n' in cols else 'driver_n'
            if wr_col in cols and n_col in cols:
                df_nc['_driver_clean'] = df_nc['driver'].str.strip().str.upper()
                jockey_merge = jockey_source.copy()
                jockey_merge['driver'] = jockey_merge['driver'].str.strip().str.upper()
                df_nc = df_nc.merge(
                    jockey_merge[['driver', wr_col, n_col]],
                    left_on='_driver_clean', right_on='driver', how='left',
                    suffixes=('','_jk'))
                df_nc['jockey_win_rate_bayes'] = df_nc[wr_col].fillna(fallback)
                df_nc['jockey_n']              = df_nc[n_col].fillna(0)
                df_nc.drop(columns=['_driver_clean','driver_jk'], errors='ignore', inplace=True)
        df_nc['jockey_win_rate_bayes'] = df_nc['jockey_win_rate_bayes'].fillna(fallback)
        df_nc['jockey_n']              = df_nc['jockey_n'].fillna(0)
        df_nc['jockey_fiable']         = (df_nc['jockey_n'] >= 5).astype(float)
        df_nc['score_jockey'] = (_norm_mix_g(df_nc['jockey_win_rate_bayes'], fallback*0.8, 0.35)*0.60 +
                                  df_nc['jockey_fiable']*0.25 +
                                  _norm_mix_g(df_nc['jockey_n'], 5, 50)*0.15).clip(0, 1)
        df_nc['score_duo'] = df_nc['score_jockey']  # alias
    else:
        # MONTE : score duo cheval+jockey classique
        if _duo_stats is not None:
            df_nc = df_nc.merge(_duo_stats[['nom','driver','duo_win_rate_bayes','duo_n']],
                                on=['nom','driver'], how='left')
        if 'duo_win_rate_bayes' not in df_nc.columns:
            df_nc['duo_win_rate_bayes'] = fallback
            df_nc['duo_n'] = 0
        df_nc['duo_win_rate_bayes'] = df_nc['duo_win_rate_bayes'].fillna(fallback)
        df_nc['duo_n']              = df_nc['duo_n'].fillna(0)
        df_nc['duo_fiable']         = (df_nc['duo_n'] >= 2).astype(float)
        df_nc['score_duo'] = (_norm_mix_g(df_nc['duo_win_rate_bayes'], fallback*0.8, 0.65)*0.60 +
                              df_nc['duo_fiable']*0.25 +
                              _norm_mix_g(df_nc['duo_n'], 1, 15)*0.15).clip(0, 1)
        df_nc['score_jockey'] = df_nc['score_duo']  # alias

    # ── Score historique ─────────────────────────────────────
    if _hist_snapshot is not None:
        hist_cols = [c for c in ['nom','hist_nb','hist_taux_top3','hist_moy_classement',
                                  'hist_tendance','hist_moy_cote'] if c in _hist_snapshot.columns]
        df_nc = df_nc.merge(_hist_snapshot[hist_cols], on='nom', how='left')
    for col, val in [('hist_nb',0),('hist_taux_top3',fallback),('hist_moy_classement',8),
                     ('hist_tendance',0),('hist_moy_cote',15)]:
        if col not in df_nc.columns: df_nc[col] = val
        df_nc[col] = df_nc[col].fillna(val)
    hist_cote_med = df_nc['hist_moy_cote'].median()
    df_nc['score_historique'] = (
        _norm_mix_g(df_nc['hist_taux_top3'], 0, 0.7)*0.35 +
        _norm_mix_g(10 - df_nc['hist_moy_classement'], -5, 9)*0.25 +
        _norm_g(df_nc['hist_nb'], 0, 20)*0.10 +
        _norm_mix_g(df_nc['hist_tendance'], -3, 3)*0.20 +
        (1 - _norm_g(df_nc['hist_moy_cote'].fillna(hist_cote_med).fillna(15), 2, 30))*0.10
    ).clip(0, 1)

    # ── Score gains ──────────────────────────────────────────
    df_nc['score_gains'] = (
        _norm_mix_g(df_nc['ratio_victoires'],  0, 0.4)*0.30 +
        _norm_mix_g(df_nc['gains_par_course'], 0, 8000)*0.25 +
        _norm_mix_g(df_nc['gains_annee'],      0, 150000)*0.20 +
        _norm_mix_g(df_nc['ratio_gains_rec'],  0, 0.5)*0.15 +
        _norm_mix_g(df_nc['ratio_places'],     0, 0.6)*0.10
    ).clip(0, 1)

    # ── Score handicap ───────────────────────────────────────
    s_hcap_val   = _norm_mix_g(df_nc['handicap_valeur'].fillna(df_nc['handicap_valeur'].median()), 30, 70)
    s_poids_rang = df_nc['rang_poids_n']
    s_poids_abs  = 1 - _norm_g(df_nc['handicap_poids'], 520, 640)
    df_nc['score_handicap'] = (s_hcap_val*0.50 + s_poids_rang*0.30 + s_poids_abs*0.20).clip(0, 1)

    # ── Score cote ───────────────────────────────────────────
    s_cote_rang   = 1 - df_nc['rang_cote_norme']
    s_ecart       = _norm_g(-df_nc['ecart_cotes'].abs(), -10, 0)
    s_cote_direct = 1 - _norm_mix_g(df_nc['rapport_direct'].fillna(df_nc['rapport_ref']), 2, 50)
    df_nc['score_cote'] = (s_cote_rang*0.50 + s_cote_direct*0.35 +
                           s_ecart*0.15).clip(0, 1).fillna(s_cote_rang.clip(0, 1))

    # ── Scoring final via XGBoost galop ──────────────────────
    bundle_galop = _models_galop[discipline_raw]
    features_5   = bundle_galop['features']
    poids_cote   = bundle_galop.get('poids_cote_fixe', 0.15)
    poids_xgb    = bundle_galop.get('poids_xgb', 0.85)

    # ── Features PLAT V8 brutes ───────────────────────────────
    if discipline_raw == 'PLAT':
        fallback_v8 = bundle_galop.get('prior_win', 0.098) * bundle_galop.get('k_bayes', 10) / (bundle_galop.get('k_bayes', 10) + 1)

        # Stats driver depuis bundle — robuste aux deux nommages (jockey_* ou driver_*)
        driver_stats = bundle_galop.get('driver_stats') or bundle_galop.get('jockey_stats')
        if driver_stats is not None:
            try:
                cols_ds = driver_stats.columns
                wr_col = 'driver_win_rate_bayes' if 'driver_win_rate_bayes' in cols_ds else ('jockey_win_rate_bayes' if 'jockey_win_rate_bayes' in cols_ds else None)
                n_col  = 'driver_n' if 'driver_n' in cols_ds else ('jockey_n' if 'jockey_n' in cols_ds else None)
                if wr_col and n_col and 'driver' in cols_ds:
                    tmp = driver_stats[['driver', wr_col, n_col]].rename(
                        columns={wr_col: 'driver_win_rate_bayes', n_col: 'driver_n'})
                    df_nc = df_nc.merge(tmp, on='driver', how='left')
            except Exception as e:
                print(f"⚠️  Galop driver_stats merge échoué ({e})")
        if 'driver_win_rate_bayes' not in df_nc.columns:
            df_nc['driver_win_rate_bayes'] = fallback_v8
        if 'driver_n' not in df_nc.columns:
            df_nc['driver_n'] = 0
        df_nc['driver_win_rate_bayes'] = df_nc['driver_win_rate_bayes'].fillna(fallback_v8)
        df_nc['driver_n']              = df_nc['driver_n'].fillna(0)

        # Stats duo depuis bundle — robuste aux deux nommages
        duo_stats = bundle_galop.get('duo_stats')
        if duo_stats is not None:
            try:
                cols_du = duo_stats.columns
                duo_col = 'duo_win_rate_bayes' if 'duo_win_rate_bayes' in cols_du else ('duo_jockey_win_rate' if 'duo_jockey_win_rate' in cols_du else None)
                if duo_col and 'nom' in cols_du and 'driver' in cols_du:
                    tmp = duo_stats[['nom','driver', duo_col]].rename(columns={duo_col: 'duo_win_rate_bayes'})
                    df_nc = df_nc.merge(tmp, on=['nom','driver'], how='left')
            except Exception as e:
                print(f"⚠️  Galop duo_stats merge échoué ({e})")
        if 'duo_win_rate_bayes' not in df_nc.columns:
            df_nc['duo_win_rate_bayes'] = fallback_v8
        df_nc['duo_win_rate_bayes'] = df_nc['duo_win_rate_bayes'].fillna(fallback_v8)

        # Stats entraineur depuis bundle
        entr_stats = bundle_galop.get('entr_stats')
        if entr_stats is not None:
            try:
                if 'entraineur' in entr_stats.columns and 'entr_win_rate_bayes' in entr_stats.columns:
                    df_nc = df_nc.merge(entr_stats[['entraineur','entr_win_rate_bayes']],
                                        on='entraineur', how='left')
            except Exception as e:
                print(f"⚠️  Galop entr_stats merge échoué ({e})")
        if 'entr_win_rate_bayes' not in df_nc.columns:
            df_nc['entr_win_rate_bayes'] = fallback_v8
        df_nc['entr_win_rate_bayes'] = df_nc['entr_win_rate_bayes'].fillna(fallback_v8)

        # Spécialisation distance
        df_nc['tranche_dist'] = pd.cut(df_nc['distance'],
            bins=[0,1600,2100,2700,9999], labels=['court','moyen','long','tres_long']).astype(str)
        spec_dist = bundle_galop.get('spec_dist')
        if spec_dist is not None:
            spec = spec_dist.copy()
            if 'tranche_distance' in spec.columns:
                spec = spec.rename(columns={'tranche_distance':'tranche_dist'})
            df_nc = df_nc.merge(spec[['nom','tranche_dist','spec_dist_rate']],
                                on=['nom','tranche_dist'], how='left')
        df_nc['spec_dist_rate'] = df_nc.get('spec_dist_rate', pd.Series([fallback_v8]*len(df_nc))).fillna(fallback_v8)

        # Features numériques
        df_nc['log_montant_prix'] = np.log1p(df_nc['montant_prix'])
        df_nc['nb_partants_c']    = len(df_nc)

        # Stalle de départ
        df_nc['place_corde_norm'] = df_nc['place_corde'].replace(0, np.nan).fillna(len(df_nc)/2)
        df_nc['stalle_avantage']  = df_nc['place_corde'].apply(
            lambda x: 1.0 if 3<=x<=5 else (0.7 if x in [1,2,6,7] else 0.3)
            if pd.notna(x) and x > 0 else 0.5
        )

        # Terrain
        _terrain_map = {
            'Lourd': 1.0, 'Collant': 1.5, 'Très souple': 2.0,
            'Souple': 2.5, 'Bon souple': 3.0, 'Bon léger': 3.5,
            'Bon': 4.0, 'Léger': 4.5,
            'PSF LENTE': 3.0, 'PSF': 3.5, 'PSF STANDARD': 3.5,
            'PSF RAPIDE': 4.0, 'PSF TRES RAPIDE': 4.5,
        }
        terrain_label = str(df_nc['terrain_label'].iloc[0]) if 'terrain_label' in df_nc.columns and len(df_nc) > 0 else ''
        df_nc['terrain_num'] = _terrain_map.get(terrain_label, bundle_galop.get('median_terrain_num', 3.5))
        df_nc['terrain_psf'] = 1 if 'PSF' in terrain_label else 0

    df_input = pd.DataFrame(index=df_nc.index)
    for feat in features_5:
        df_input[feat] = df_nc[feat] if feat in df_nc.columns else bundle_galop.get(f'median_{feat}', 0.5)

    # Détection automatique XGBoost (sklearn API) vs LightGBM Booster
    _mdl = bundle_galop['model']
    if hasattr(_mdl, 'predict_proba'):
        probas = _mdl.predict_proba(df_input[features_5])[:, 1]
    else:
        # LightGBM Booster ou XGBoost Booster : predict() retourne directement les scores
        probas = np.asarray(_mdl.predict(df_input[features_5]))

    # PLAT V8 : pipeline brut sans score_cote (comme ATTELÉ)
    if discipline_raw == 'PLAT':
        proba_min   = bundle_galop.get('proba_min', 0.19)
        proba_max   = bundle_galop.get('proba_max', 0.70)
        score_final = pd.Series(probas, index=df_nc.index)
        df_nc['note_pmu']  = np.round((score_final - proba_min) / (proba_max - proba_min) * 19 + 1).clip(1, 20).astype(int)
        df_nc['proba_pmu'] = score_final
    else:
        # HAIE/MONTE : pipeline brut sans score_cote (comme ATTELÉ et PLAT)
        poids_cote  = bundle_galop.get('poids_cote_fixe', 0.0)   # fallback 0.0 — cote désactivée
        poids_xgb   = bundle_galop.get('poids_xgb', 1.0)         # fallback 1.0
        if poids_cote > 0:
            score_final = pd.Series(poids_xgb * probas + poids_cote * df_nc['score_cote'].values,
                                    index=df_nc.index)
        else:
            score_final = pd.Series(probas, index=df_nc.index)
        g_min = bundle_galop.get('proba_min')
        g_max = bundle_galop.get('proba_max')
        df_nc['note_pmu']  = _proba_to_note_v7(score_final, proba_min_ref=g_min, proba_max_ref=g_max)
        df_nc['proba_pmu'] = score_final

    # ── Résultat JSON ─────────────────────────────────────────
    result = []
    for _, row in df_nc.sort_values('note_pmu', ascending=False).iterrows():
        result.append({
            "numero":    int(row['numero']),
            "nom":       str(row['nom']),
            "note_pmu":  int(row['note_pmu']),
            "proba_pmu": round(float(row['proba_pmu']) * 100, 1) if pd.notna(row['proba_pmu']) else 0,
            "driver":    str(row['driver']),
            "cote":      float(row['_cote_app']) if pd.notna(row['_cote_app']) else None,
            "avis":      int(row['avis_entraineur']) if pd.notna(row.get('avis_entraineur')) else 0,
            "scores": {
                "forme":      int(round(float(row['score_forme'])      * 100)) if pd.notna(row['score_forme'])      else 0,
                "duo":        int(round(float(row['score_duo'])        * 100)) if pd.notna(row['score_duo'])        else 0,
                "jockey":     int(round(float(row['score_jockey'])     * 100)) if pd.notna(row.get('score_jockey')) else 0,
                "historique": int(round(float(row['score_historique']) * 100)) if pd.notna(row['score_historique']) else 0,
                "gains":      int(round(float(row['score_gains'])      * 100)) if pd.notna(row['score_gains'])      else 0,
                "handicap":   int(round(float(row['score_handicap'])   * 100)) if pd.notna(row['score_handicap'])   else 0,
                "cote":       int(round(float(row['score_cote'])       * 100)) if pd.notna(row['score_cote'])       else 0,
            },
            "taux_disq":    round(float(row['mus_taux_disq']) * 100, 1) if pd.notna(row.get('mus_taux_disq')) else 0,
            "musique":      str(row.get('musique', '')) if row.get('musique') else '',
            "handicap_poids":  int(row.get('handicap_poids', 0)),
            "handicap_valeur": float(row.get('handicap_valeur', 0)),
        })

    return jsonify({
        "date":       date_str,
        "reunion":    r_num,
        "course":     c_num,
        "discipline": discipline_raw,
        "version":    f"v2_galop_{discipline_raw.lower()}",
        "chevaux":    result,
    })


@app.route('/notes_pmu', methods=['GET'])
def notes_pmu():
    """
    Calcule les notes PMU pour une course donnée.
    Paramètres GET : date (DDMMYYYY), reunion (int), course (int)
    Ex: /notes_pmu?date=05032026&reunion=1&course=5
    """
    if _model_pmu is None:
        return jsonify({"error": "Modèle PMU non disponible"}), 503

    date_str = request.args.get('date', '')
    r_num    = request.args.get('reunion', '')
    c_num    = request.args.get('course', '')

    if not date_str or not r_num or not c_num:
        return jsonify({"error": "Paramètres requis : date, reunion, course"}), 400
    try:
        r_num = int(r_num); c_num = int(c_num)
    except ValueError:
        return jsonify({"error": "reunion et course doivent être des entiers"}), 400

    # ── Conditions de course & performances détaillées ───────
    conditions    = _fetch_conditions(date_str, r_num, c_num)
    perfs_map     = _fetch_performances(date_str, r_num, c_num)
    discipline_raw = conditions.get('discipline_raw', '')

    # ── Routing par discipline ────────────────────────────────
    # STEEPLECHASE et CROSS : non supportés
    if discipline_raw in DISCIPLINES_SKIP:
        return jsonify({"error": f"Discipline {discipline_raw} non supportée"}), 400
    # Galop : PLAT, HAIE, MONTE
    is_galop = discipline_raw in DISCIPLINES_GALOP
    if is_galop and discipline_raw not in _models_galop:
        return jsonify({"error": f"Modèle {discipline_raw} non disponible"}), 503

    # ── Participants ──────────────────────────────────────────
    # Stratégie : offline (stable) avec fallback online (pour avisEntraineur)
    participants = []
    _avis_map_raw = {}
    # Tentative 1 — online avec avecOrdreArrivee=false (avisEntraineur disponible)
    try:
        url_online = (f"https://online.turfinfo.api.pmu.fr/rest/client/61/programme"
                      f"/{date_str}/R{r_num}/C{c_num}/participants"
                      f"?specialisation=INTERNET&avecOrdreArrivee=false")
        resp_on = http_requests.get(url_online, timeout=6)
        if resp_on.status_code == 200:
            participants = resp_on.json().get('participants', [])
            _avis_map_raw = {str(p.get('numPmu','')): p.get('avisEntraineur','NEUTRE')
                             for p in participants}
    except Exception:
        pass
    # Tentative 2 — offline (plus stable) si online a échoué
    if not participants:
        try:
            url_off = (f"https://offline.turfinfo.api.pmu.fr/rest/client/7/programme"
                       f"/{date_str}/R{r_num}/C{c_num}/participants")
            resp_off = http_requests.get(url_off, timeout=8)
            resp_off.raise_for_status()
            participants = resp_off.json().get('participants', [])
        except Exception as e:
            return jsonify({"error": f"Erreur API PMU : {str(e)}"}), 502

    if not participants:
        return jsonify({"error": "Aucun participant trouvé"}), 404

    # Médiane rapport de référence
    rapports_course = [
        p['dernierRapportReference'].get('rapport')
        for p in participants
        if p.get('dernierRapportReference') and p.get('statut') != 'NON_PARTANT'
    ]
    rapports_course  = [r for r in rapports_course if r]
    mediane_rr       = float(np.median(rapports_course)) if rapports_course else _mediane_rapport_ref

    rows = []
    for p in participants:
        if p.get('statut') == 'NON_PARTANT' or p.get('incident') == 'NON_PARTANT':
            continue
        mus        = _parser_musique_api(p.get('musique', ''))
        musique_brute = p.get('musique', '')
        gains      = p.get('gainsParticipant', {}) or {}
        rk         = p.get('reductionKilometrique', 0) or 0
        num_pmu    = p.get('numPmu')
        nb_courses = p.get('nombreCourses', 0) or 0
        driver_nom = (p.get('driver', {}).get('nom', '')
                      if isinstance(p.get('driver'), dict) else str(p.get('driver', '')))
        entr_nom   = (p.get('entraineur', {}).get('nom', '')
                      if isinstance(p.get('entraineur'), dict) else str(p.get('entraineur', '')))
        nb_victoires = p.get('nombreVictoires', 0) or 0
        nb_places    = p.get('nombrePlaces', 0) or 0
        gains_car    = gains.get('gainsCarriere', 0) or 0
        gains_ann    = gains.get('gainsAnneeEnCours', 0) or 0

        rapport_ref = None
        if p.get('dernierRapportReference'):
            rapport_ref = p['dernierRapportReference'].get('rapport')
        if rapport_ref is None:
            rapport_ref = mediane_rr

        # Cote en temps réel (rapport direct)
        cote_app = None
        if p.get('dernierRapportDirect'):
            cote_app = p['dernierRapportDirect'].get('rapport')
        if cote_app is None and p.get('dernierRapportReference'):
            cote_app = p['dernierRapportReference'].get('rapport')

        perf = perfs_map.get(num_pmu, _perf_vide())

        row = {
            'numero':            num_pmu,
            'nom':               p.get('nom', ''),
            # Conditions course
            'distance':          conditions['distance'],
            'montant_prix':      conditions['montant_prix'],
            'discipline':        conditions['discipline'],
            'corde':             conditions['corde'],
            'condition_sexe':    conditions['condition_sexe'],
            'nb_partants':       conditions['nb_partants'],
            # Cheval
            'age':               p.get('age', 0) or 0,
            'deferre':           _ferrage_map_pmu.get(p.get('deferre', 'FERRE'), 0),
            'oeilleres':         1 if p.get('oeilleres') else 0,
            'driver':            driver_nom,
            'entraineur':        entr_nom,
            'nb_courses':        nb_courses,
            'nb_victoires':      nb_victoires,
            'nb_places':         nb_places,
            'gains_carriere':    gains_car,
            'gains_annee':       gains_ann,
            'reduction_km_corr': rk if rk > 0 else 72600,
            'avis_entraineur':   _avis_map_pmu.get(
                                    _avis_map_raw.get(str(p.get('numPmu','')),
                                    p.get('avisEntraineur', 'NEUTRE')), 0),
            'rapport_ref':       float(rapport_ref),
            'rapport_direct':    float(cote_app) if cote_app else float(rapport_ref),
            'ecart_cotes':       float(cote_app - rapport_ref) if cote_app else 0.0,
            'log_rapport_ref':   float(np.log1p(rapport_ref)),
            'nb_places_second':  p.get('nombrePlacesSecond', 0) or 0,
            'nb_places_troisieme': p.get('nombrePlacesTroisieme', 0) or 0,
            'temps_obtenu':      float(p.get('tempsObtenu', 0) or 0),
            'handicap_distance': float(p.get('handicapDistance', 0) or conditions['distance'] or 0),
            'handicap_poids':    float(p.get('handicapPoids', 0) or 0),
            'handicap_valeur':   float(p.get('handicapValeur', 0) or 0),
            '_cote_app':         cote_app,
            'place_corde':       float(p.get('placeCorde', 0) or 0),
            'nb_jours_absence':  (lambda ts: max(0, (pd.Timestamp.now() - pd.Timestamp(ts/1000, unit='s')).days)
                                  if ts else 30)(
                                  (p.get('dernierRapportDirect') or {}).get('dateRapport') or
                                  (p.get('dernierRapportReference') or {}).get('dateRapport')),
            'terrain_label':     conditions.get('terrain_label', ''),
            'terrain_val':       float(conditions.get('terrain_val', 3.0)),
            'hippodrome_code':   conditions.get('hippodrome_code', ''),
            'hippodrome_nom':    conditions.get('hippodrome_nom', ''),
        }
        row['musique'] = musique_brute
        row.update(mus)
        row.update(perf)
        rows.append(row)

    df_nc = pd.DataFrame(rows)

    # Features de base toujours nécessaires
    df_nc['nb_partants_c'] = len(df_nc)
    df_nc['sexe']          = df_nc['condition_sexe'] if 'condition_sexe' in df_nc.columns else 2
    df_nc['log_montant_prix'] = np.log1p(df_nc['montant_prix'])

    # ════════════════════════════════════════════════════════════
    # ROUTING GALOP — si discipline galop, pipeline dédié
    # ════════════════════════════════════════════════════════════
    if is_galop:
        return _notes_pmu_galop(df_nc, discipline_raw, date_str, r_num, c_num)

    # ════════════════════════════════════════════════════════════
    # ARCHITECTURE V6 — DEUX ÉTAPES
    # Étape 1 : scores métier déterministes (6 dimensions)
    # Étape 2 : XGBoost sur ces 6 scores uniquement
    # ════════════════════════════════════════════════════════════

    # ── Features dérivées brutes (inchangées) ─────────────────
    df_nc['ratio_victoires']        = df_nc['nb_victoires'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places']           = df_nc['nb_places']    / (df_nc['nb_courses'] + 1)
    df_nc['gains_par_course']       = df_nc['gains_carriere'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_gains_rec']        = df_nc['gains_annee'] / (df_nc['gains_carriere'] + 1)
    df_nc['ratio_places_second']    = df_nc['nb_places_second']    / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places_troisieme'] = df_nc['nb_places_troisieme'] / (df_nc['nb_courses'] + 1)
    df_nc['temps_norme'] = df_nc.apply(
        lambda r: round(r['temps_obtenu'] / r['handicap_distance'], 4)
        if r['handicap_distance'] > 0 and r['temps_obtenu'] > 0 else np.nan, axis=1
    )
    df_nc['log_distance']     = np.log1p(df_nc['distance'])
    df_nc['log_montant_prix'] = np.log1p(df_nc['montant_prix'])
    df_nc['rang_cote_course'] = df_nc['rapport_ref'].rank(ascending=True, method='min')
    nb_ch = len(df_nc)
    df_nc['rang_cote_norme']  = (df_nc['rang_cote_course'] - 1) / (nb_ch - 1 + 1e-8)
    df_nc['tranche_distance'] = pd.cut(
        df_nc['distance'], bins=[0, 1600, 2100, 2700, 9999],
        labels=['court', 'moyen', 'long', 'tres_long']
    ).astype(str)

    _fallback = _prior_pmu * _k_bayes_pmu / (_k_bayes_pmu + 1) if _prior_pmu else 0.094

    # ── Driver stats ──────────────────────────────────────────
    if _le_driver is not None and _driver_stats is not None:
        top_drivers = set(_le_driver.classes_)
        df_nc['driver_enc'] = df_nc['driver'].apply(lambda x: x if x in top_drivers else 'AUTRE')
        df_nc['driver_id']  = _le_driver.transform(df_nc['driver_enc'])
        d_cols = ['driver', 'driver_win_rate_bayes', 'driver_n']
        if 'driver_place_rate_bayes' in _driver_stats.columns:
            d_cols += ['driver_place_rate_bayes', 'driver_disq']
        df_nc = df_nc.merge(_driver_stats[d_cols], on='driver', how='left')
        df_nc['driver_win_rate_bayes']   = df_nc['driver_win_rate_bayes'].fillna(_fallback)
        df_nc['driver_n']                = df_nc['driver_n'].fillna(0)
        if 'driver_place_rate_bayes' in df_nc.columns:
            df_nc['driver_place_rate_bayes'] = df_nc['driver_place_rate_bayes'].fillna(_fallback)
            df_nc['driver_disq']             = df_nc['driver_disq'].fillna(0)
    elif _driver_stats is not None:
        df_nc = df_nc.merge(
            _driver_stats[['driver', 'driver_win_rate_bayes', 'driver_n']],
            on='driver', how='left')
    if 'driver_win_rate_bayes' not in df_nc.columns:
        df_nc['driver_win_rate_bayes'] = _fallback
        df_nc['driver_n']              = 0
    df_nc['driver_win_rate_bayes'] = df_nc['driver_win_rate_bayes'].fillna(_fallback)
    df_nc['driver_n']              = df_nc['driver_n'].fillna(0)

    if _le_entr is not None:
        top_entrs = set(_le_entr.classes_)
        df_nc['entraineur_enc'] = df_nc['entraineur'].apply(lambda x: x if x in top_entrs else 'AUTRE')
        df_nc['entraineur_id']  = _le_entr.transform(df_nc['entraineur_enc'])
    if _entr_stats is not None:
        df_nc = df_nc.merge(
            _entr_stats[['entraineur', 'entr_win_rate_bayes', 'entr_n']],
            on='entraineur', how='left')
    if 'entr_win_rate_bayes' not in df_nc.columns:
        df_nc['entr_win_rate_bayes'] = _fallback
        df_nc['entr_n'] = 0
    df_nc['entr_win_rate_bayes'] = df_nc['entr_win_rate_bayes'].fillna(_fallback)
    df_nc['entr_n']              = df_nc['entr_n'].fillna(0)

    if _duo_stats is not None:
        df_nc = df_nc.merge(
            _duo_stats[['nom', 'driver', 'duo_win_rate_bayes', 'duo_n']],
            on=['nom', 'driver'], how='left')
    if 'duo_win_rate_bayes' not in df_nc.columns:
        df_nc['duo_win_rate_bayes'] = _fallback
        df_nc['duo_n'] = 0
    df_nc['duo_win_rate_bayes'] = df_nc['duo_win_rate_bayes'].fillna(_fallback)
    df_nc['duo_n']              = df_nc['duo_n'].fillna(0)
    df_nc['duo_fiable']         = (df_nc['duo_n'] >= 2).astype(int)
    df_nc['duo_fiable_v2']      = (df_nc['duo_n'] >= _duo_fiable_seuil_v9).astype(int)

    if _spec_dist is not None:
        spec_cols_dist = 'tranche_dist' if 'tranche_dist' in _spec_dist.columns else 'tranche_distance'
        df_nc = df_nc.merge(
            _spec_dist[['nom', spec_cols_dist, 'spec_dist_rate']].rename(
                columns={spec_cols_dist: 'tranche_distance'}),
            on=['nom', 'tranche_distance'], how='left')
    if 'spec_dist_rate' not in df_nc.columns:
        df_nc['spec_dist_rate'] = _fallback
        df_nc['spec_n'] = 0
    df_nc['spec_dist_rate'] = df_nc['spec_dist_rate'].fillna(_fallback)
    df_nc['spec_n']         = df_nc['spec_n'].fillna(0) if 'spec_n' in df_nc.columns else 0

    if _spec_disc is not None:
        df_nc = df_nc.merge(
            _spec_disc[['nom', 'discipline', 'spec_disc_rate']],
            on=['nom', 'discipline'], how='left')
    if 'spec_disc_rate' not in df_nc.columns:
        df_nc['spec_disc_rate'] = _fallback
    df_nc['spec_disc_rate'] = df_nc['spec_disc_rate'].fillna(_fallback)

    if _hist_snapshot is not None:
        hist_cols_dispo = [c for c in ['nom', 'hist_nb', 'hist_moy_classement', 'hist_nb_top3',
                            'hist_taux_top3', 'hist_moy_temps', 'hist_tendance', 'hist_moy_cote',
                            'courses_60j']
                           if c in _hist_snapshot.columns]
        df_nc = df_nc.merge(_hist_snapshot[hist_cols_dispo], on='nom', how='left')
    if 'courses_60j' not in df_nc.columns:
        df_nc['courses_60j'] = 0
    df_nc['courses_60j'] = df_nc['courses_60j'].fillna(0).astype(int)
    for col in ['hist_nb', 'hist_nb_top3']:
        if col not in df_nc.columns:
            df_nc[col] = 0
        df_nc[col] = df_nc[col].fillna(0)
    for col in ['hist_moy_classement', 'hist_taux_top3', 'hist_moy_temps',
                'hist_tendance', 'hist_moy_cote']:
        if col not in df_nc.columns:
            df_nc[col] = np.nan

    # ── Features V9 : forme récente & momentum duo ────────────
    prior_v9 = _prior_pmu if _prior_pmu else 0.309

    # ── duo_momentum_3 ────────────────────────────────────────
    try:
        if _duo_momentum_snap is not None:
            snap = _duo_momentum_snap.copy().reset_index(drop=True)
            # Normaliser les colonnes selon le format du groupby
            if 'duo_momentum_3' not in snap.columns:
                # Format groupby multi-index : (nom, driver) → valeur
                snap = snap.rename(columns={snap.columns[-1]: 'duo_momentum_3'})
            if 'nom' in snap.columns and 'driver' in snap.columns and 'duo_momentum_3' in snap.columns:
                df_nc = df_nc.merge(
                    snap[['nom', 'driver', 'duo_momentum_3']],
                    on=['nom', 'driver'], how='left')
    except Exception as e:
        print(f"⚠️  duo_momentum_snap merge échoué ({e}) — fallback")
    if 'duo_momentum_3' not in df_nc.columns:
        df_nc['duo_momentum_3'] = prior_v9
    df_nc['duo_momentum_3'] = df_nc['duo_momentum_3'].fillna(prior_v9)

    # ── top3_3courses ─────────────────────────────────────────
    try:
        if _top3_3courses_snap is not None:
            snap3 = _top3_3courses_snap.copy().reset_index(drop=True)
            if 'top3_3courses' not in snap3.columns:
                snap3 = snap3.rename(columns={snap3.columns[-1]: 'top3_3courses'})
            if 'nom' in snap3.columns and 'top3_3courses' in snap3.columns:
                df_nc = df_nc.merge(
                    snap3[['nom', 'top3_3courses']],
                    on='nom', how='left')
    except Exception as e:
        print(f"⚠️  top3_3courses_snap merge échoué ({e}) — fallback")
    if 'top3_3courses' not in df_nc.columns:
        df_nc['top3_3courses'] = prior_v9
    df_nc['top3_3courses'] = df_nc['top3_3courses'].fillna(prior_v9)

    # ── top3_60j ──────────────────────────────────────────────
    try:
        if _top3_60j_snap is not None:
            snap60 = _top3_60j_snap.copy().reset_index(drop=True)
            # Identifier la colonne top3_60j (peut avoir des colonnes extra)
            col_60j = next((c for c in snap60.columns
                            if 'top3_60j' in c or '60j' in c), None)
            if col_60j and col_60j != 'top3_60j':
                snap60 = snap60.rename(columns={col_60j: 'top3_60j'})
            # Garder uniquement nom + top3_60j pour éviter les conflits
            cols_ok = [c for c in ['nom', 'top3_60j'] if c in snap60.columns]
            if len(cols_ok) == 2:
                snap60 = snap60[cols_ok].drop_duplicates(subset=['nom'])
                df_nc = df_nc.merge(snap60, on='nom', how='left')
    except Exception as e:
        print(f"⚠️  top3_60j_snap merge échoué ({e}) — fallback")
    if 'top3_60j' not in df_nc.columns:
        df_nc['top3_60j'] = prior_v9
    df_nc['top3_60j'] = df_nc['top3_60j'].fillna(prior_v9)

    # ════════════════════════════════════════════════════════════
    # FEATURES V10 — fraicheur, progression, place_avantage,
    #                aptitude_piste
    # ════════════════════════════════════════════════════════════

    # ── fraicheur_score ───────────────────────────────────────
    # On ne peut pas calculer les jours depuis la dernière course
    # en temps réel sans l'historique — on utilise le snapshot
    try:
        if _fraicheur_snap is not None:
            snap_f = _fraicheur_snap.copy().reset_index(drop=True)
            if 'nom' in snap_f.columns and 'fraicheur_jours' in snap_f.columns:
                df_nc = df_nc.merge(snap_f[['nom','fraicheur_jours']],
                                    on='nom', how='left')
    except Exception as e:
        print(f"⚠️  fraicheur_snap merge échoué ({e})")
    if 'fraicheur_jours' not in df_nc.columns:
        df_nc['fraicheur_jours'] = 21.0   # fallback = zone optimale
    df_nc['fraicheur_jours'] = df_nc['fraicheur_jours'].fillna(21.0)

    def _fraicheur_score(j):
        if j <= 7:    return 0.3
        elif j <= 14: return 0.8
        elif j <= 21: return 1.0
        elif j <= 30: return 0.9
        elif j <= 45: return 0.7
        elif j <= 60: return 0.5
        else:         return 0.3
    df_nc['fraicheur_score'] = df_nc['fraicheur_jours'].apply(_fraicheur_score)

    # ── progression_norm ──────────────────────────────────────
    try:
        if _progression_snap is not None:
            snap_p = _progression_snap.copy().reset_index(drop=True)
            if 'nom' in snap_p.columns and 'progression_norm' in snap_p.columns:
                df_nc = df_nc.merge(snap_p[['nom','progression_norm']],
                                    on='nom', how='left')
    except Exception as e:
        print(f"⚠️  progression_snap merge échoué ({e})")
    if 'progression_norm' not in df_nc.columns:
        df_nc['progression_norm'] = 0.0   # fallback = neutre
    df_nc['progression_norm'] = df_nc['progression_norm'].fillna(0.0)

    # ── place_avantage — calibré sur données réelles (courbe en cloche) ─
    _PA_MAP = {
        1:0.797, 2:0.862, 3:0.900, 4:0.982, 5:1.001,
        6:0.959, 7:0.958, 8:0.911, 9:0.828, 10:0.791,
        11:0.810, 12:0.808, 13:0.753, 14:0.779, 15:0.742,
        16:0.720, 17:0.763, 18:0.823
    }
    def _avantage_numero(n):
        try: return _PA_MAP.get(int(n), 0.75)
        except: return 0.75
    df_nc['place_avantage'] = df_nc['numero'].apply(_avantage_numero)

    # ── aptitude_piste ────────────────────────────────────────
    try:
        if _aptitude_snap is not None:
            snap_a = _aptitude_snap.copy().reset_index(drop=True)
            # Colonne tranche peut s'appeler tranche_distance
            if 'tranche_distance' in snap_a.columns and \
               'aptitude_piste' in snap_a.columns and \
               'nom' in snap_a.columns:
                df_nc = df_nc.merge(
                    snap_a[['nom','tranche_distance','aptitude_piste']],
                    on=['nom','tranche_distance'], how='left')
    except Exception as e:
        print(f"⚠️  aptitude_snap merge échoué ({e})")
    if 'aptitude_piste' not in df_nc.columns:
        df_nc['aptitude_piste'] = prior_v9
    df_nc['aptitude_piste'] = df_nc['aptitude_piste'].fillna(prior_v9)

    # ════════════════════════════════════════════════════════════
    # FEATURES V12 — chrono_norm_peloton, ratio_niveau,
    #                driver_win_rate_90j, entr_win_rate_30j
    # ════════════════════════════════════════════════════════════

    # ── chrono_norm_peloton — RK normalisé dans le peloton ────
    # Feature RELATIVE : calculée sur le peloton courant
    # RK bas = rapide = bon → inverser pour que 1.0 = meilleur
    rk_vals = df_nc['reduction_km_corr'].copy()
    # Remplacer les valeurs manquantes par la médiane du peloton
    rk_med  = rk_vals[(rk_vals > 0) & (rk_vals < 100000)].median()
    if pd.isna(rk_med):
        rk_med = 76100
    rk_vals = rk_vals.where((rk_vals > 0) & (rk_vals < 100000), rk_med)
    rk_min, rk_max = rk_vals.min(), rk_vals.max()
    if rk_max - rk_min > 1:
        df_nc['chrono_norm_peloton'] = (1 - (rk_vals - rk_min) / (rk_max - rk_min)).clip(0, 1)
    else:
        df_nc['chrono_norm_peloton'] = 0.5

    # ── ratio_niveau et descente_niveau ──────────────────────
    try:
        if _niveau_snap is not None:
            snap_n = _niveau_snap.copy().reset_index(drop=True)
            if 'nom' in snap_n.columns and 'niveau_habituel' in snap_n.columns:
                df_nc = df_nc.merge(snap_n[['nom','niveau_habituel']],
                                    on='nom', how='left')
    except Exception as e:
        print(f"⚠️  niveau_snap merge échoué ({e})")
    if 'niveau_habituel' not in df_nc.columns:
        df_nc['niveau_habituel'] = df_nc['montant_prix']
    df_nc['niveau_habituel'] = df_nc['niveau_habituel'].fillna(df_nc['montant_prix'])
    df_nc['ratio_niveau']    = (df_nc['montant_prix'] /
                                 (df_nc['niveau_habituel'] + 1)).clip(0, 5)
    df_nc['descente_niveau'] = (df_nc['ratio_niveau'] < 0.8).astype(float)

    # ── driver_win_rate_90j ───────────────────────────────────
    if _driver_stats is not None and 'driver_win_rate_90j' in _driver_stats.columns:
        try:
            df_nc = df_nc.merge(
                _driver_stats[['driver','driver_win_rate_90j']],
                on='driver', how='left')
        except Exception as e:
            print(f"⚠️  driver_win_rate_90j merge échoué ({e})")
    if 'driver_win_rate_90j' not in df_nc.columns:
        df_nc['driver_win_rate_90j'] = _prior_pmu * 10 / 11 if _prior_pmu else 0.28
    df_nc['driver_win_rate_90j'] = df_nc['driver_win_rate_90j'].fillna(
        _prior_pmu * 10 / 11 if _prior_pmu else 0.28)

    # ── entr_win_rate_30j ─────────────────────────────────────
    if _entr_stats is not None and 'entr_win_rate_30j' in _entr_stats.columns:
        try:
            df_nc = df_nc.merge(
                _entr_stats[['entraineur','entr_win_rate_30j']],
                on='entraineur', how='left')
        except Exception as e:
            print(f"⚠️  entr_win_rate_30j merge échoué ({e})")
    if 'entr_win_rate_30j' not in df_nc.columns:
        df_nc['entr_win_rate_30j'] = _prior_pmu * 10 / 11 if _prior_pmu else 0.28
    df_nc['entr_win_rate_30j'] = df_nc['entr_win_rate_30j'].fillna(
        _prior_pmu * 10 / 11 if _prior_pmu else 0.28)

    # ════════════════════════════════════════════════════════════
    # FEATURES V13 — deferre_4, rk_course_norm, corde_avantage
    # ════════════════════════════════════════════════════════════

    # ── deferre_4 — déferré 4 membres ────────────────────────
    # L'API PMU retourne p.get('deferre') avec des valeurs comme
    # FERRE, DEFERRE_ANTERIEURS, DEFERRE_POSTERIEURS, DEFERRE_4_MEMBRES
    # On l'encode déjà dans _ferrage_map_pmu au chargement
    # deferre=3 dans l'historique = DEFERRE_4_MEMBRES
    if 'deferre' in df_nc.columns:
        df_nc['deferre_4'] = (df_nc['deferre'] == 3).astype(float)
        # Catégorie ferrure pour chrono filtré
        # deferre peut être un entier (0=ferré,1=partiel,2=partiel,3=total)
        # ou une string PMU (FERRE, DEFERRE_ANTERIEURS, DEFERRE_ANTERIEURS_POSTERIEURS...)
        def _get_cat_ferrure(f):
            # Format entier (encodé par _ferrage_map_pmu)
            if isinstance(f, (int, float)):
                fi = int(f)
                if fi == 3: return 'DEFERRE_TOTAL'
                if fi in [1, 2]: return 'DEFERRE_PARTIEL'
                return 'FERRE'
            # Format string PMU
            f = str(f)
            if 'ANTERIEURS_POSTERIEURS' in f and 'PROTEGE' not in f: return 'DEFERRE_TOTAL'
            if 'DEFERRE' in f: return 'DEFERRE_PARTIEL'
            return 'FERRE'
        df_nc['cat_ferrure'] = df_nc['deferre'].apply(_get_cat_ferrure)
    else:
        df_nc['deferre_4'] = 0.0

    # ── rk_course_norm — chrono normalisé par tranche ─────────
    # En production on n'a pas le chrono de la course (pas encore jouée)
    # On utilise reduction_km_v2 normalisé dans le peloton
    # (identique à chrono_norm_peloton — feature relative au peloton)
    df_nc['rk_course_norm'] = df_nc['chrono_norm_peloton']

    # ── corde_avantage — numéro de corde ─────────────────────
    # place_corde est déjà dans df_nc depuis l'API PMU
    def _avantage_corde(c):
        try:
            c = float(c)
            if c <= 0:    return 0.5   # inconnu
            elif c <= 2:  return 1.0
            elif c <= 4:  return 0.8
            elif c <= 7:  return 0.5
            elif c <= 10: return 0.3
            else:         return 0.1
        except:
            return 0.5
    if 'place_corde' in df_nc.columns:
        df_nc['corde_avantage'] = df_nc['place_corde'].apply(_avantage_corde)
    else:
        df_nc['corde_avantage'] = 0.5

    # ════════════════════════════════════════════════════════════
    # FEATURES V14 — relatives au peloton (calculées en temps réel)
    # Ces features capturent la hiérarchie dans CETTE course
    # ════════════════════════════════════════════════════════════

    # Nécessite reduction_km_v2 — calculé juste après
    # On le calcule d'abord pour les features peloton
    # Priorité : 1) chrono PMU live  2) chrono_cache historique  3) fallback
    def _get_rk_v2_val(row):
        rk = row.get('reduction_km_corr', 0)
        if rk and rk > 0 and rk != 72600:
            return rk
        # Lookup dans le cache historique (meilleur chrono des 3 dernières courses)
        nom = str(row.get('nom', '')).upper().strip()
        rk_entry = _chrono_cache.get(nom)
        if rk_entry:
            rk_hist = rk_entry['min'] if isinstance(rk_entry, dict) else float(rk_entry)
            if 60000 < rk_hist < 90000:
                return rk_hist
        # Fallback seulement si vraiment aucun chrono disponible
        return _fallback_rk_v9.get(str(row.get('tranche_distance', 'long')), 76100)
    df_nc['reduction_km_v2'] = df_nc.apply(_get_rk_v2_val, axis=1)
    df_nc['reduction_km']    = df_nc['reduction_km_v2']

    # ── reduction_km_v2_ferrure — chrono filtré par ferrure du jour ──────
    def _get_rk_ferrure(row):
        nom = str(row.get('nom', '')).upper().strip()
        ferrure = row.get('cat_ferrure', 'FERRE')
        key = f"{nom}||{ferrure}"
        entry = _chrono_cache_ferrure.get(key)
        if entry and isinstance(entry, dict):
            hist = entry.get('history', [])
            valides = [r for r in hist if 60000 < r < 90000]
            if valides:
                n = len(valides)
                # Moyenne pondérée : plus récent = poids plus élevé
                poids = [n - i for i in range(n)]
                total = sum(poids)
                return sum(v * p for v, p in zip(valides, poids)) / total
        # Fallback sur reduction_km_v2 si pas de chrono pour cette ferrure
        return row['reduction_km_v2']
    df_nc['reduction_km_v2_ferrure'] = df_nc.apply(_get_rk_ferrure, axis=1)

    # ── En production : remplacer reduction_km_v2 par reduction_km_v2_ferrure ─
    # Le modèle a été entraîné avec reduction_km_v2 = chrono réel de la course
    # En prod, on n'a pas le chrono réel → on utilise le meilleur chrono
    # par ferrure du jour (plus pertinent que toutes ferrures confondues)
    df_nc['reduction_km_v2'] = df_nc['reduction_km_v2_ferrure']

    # ── taux_completion_ferrure — fiabilité chrono par ferrure ───────────
    # Fallback : taux ferrure → taux global toutes ferrures → prior
    def _get_completion_ferrure(row):
        nom = str(row.get('nom','')).upper().strip()
        ferrure = row.get('cat_ferrure','FERRE')
        key = f"{nom}||{ferrure}"
        # 1 — Taux par ferrure spécifique
        entry = _chrono_cache_ferrure.get(key)
        if entry and isinstance(entry, dict):
            hist = entry.get('history', [])
            if len(hist) > 0:
                return sum(1 for r in hist if 60000 < r < 90000) / len(hist)
        # 2 — Fallback : taux toutes ferrures
        entry_g = _chrono_cache.get(nom)
        if entry_g and isinstance(entry_g, dict):
            hist_g = entry_g.get('history', [])
            if len(hist_g) > 0:
                return sum(1 for r in hist_g if 60000 < r < 90000) / len(hist_g)
        return 0.77  # prior global ATTELÉ (~77% courses avec chrono valide)
    df_nc['taux_completion_ferrure'] = df_nc.apply(_get_completion_ferrure, axis=1)

    # ── rk_brut + flag_chrono — chrono depuis cache historique ──
    # reduction_km_corr = 72600 en production (course pas encore jouée)
    # On utilise _chrono_cache (meilleur rk des 3 dernières courses)
    def _get_rk_brut(row):
        # D'abord essayer reduction_km_corr (si dispo et valide)
        rk = row.get('reduction_km_corr', 72600)
        if rk and rk != 72600 and 60000 < rk < 90000:
            return float(rk)
        # Sinon lookup dans le cache historique
        nom = str(row.get('nom', '')).upper().strip()
        rk_entry = _chrono_cache.get(nom)
        if rk_entry:
            rk_hist = rk_entry['min'] if isinstance(rk_entry, dict) else float(rk_entry)
            if 60000 < rk_hist < 90000:
                return float(rk_hist)
        return None
    df_nc['rk_brut'] = df_nc.apply(_get_rk_brut, axis=1)

    # Médiane des vrais chronos du peloton (sans fallback)
    rk_bruts_valides = df_nc['rk_brut'].dropna()
    rk_median_peloton = float(rk_bruts_valides.median()) if len(rk_bruts_valides) >= 3 else None

    def _flag_chrono(rk, median, musique=''):
        # Bug fix : pénaliser les disqualifications récentes (3 dernières courses)
        # Si >= 2 disq dans les 3 dernières courses → flag faible peu importe le chrono
        if musique:
            import re as _re
            clean_mus = _re.sub(r'\(\d+\)', '', str(musique)).strip()
            tokens_mus = _re.findall(r'[0-9DATRdat][amphsc]', clean_mus)[:3]  # 3 dernières
            nb_disq_recent = sum(1 for t in tokens_mus if t[0].upper() == 'D')
            if nb_disq_recent >= 2:
                return 'faible'  # trop de disq récentes → faible peu importe chrono
        if rk is None:    return 'inconnu'
        if median is None: return 'ok'
        if rk > median + 1500: return 'faible'
        if rk < median - 1500: return 'fort'
        return 'ok'

    df_nc['flag_chrono'] = df_nc.apply(
        lambda row: _flag_chrono(row['rk_brut'], rk_median_peloton,
                                  row.get('musique', '')), axis=1)

    # ── tendance_chrono — progression/dégradation ─────────────
    def _tendance_chrono(nom, musique=''):
        # Bug fix : si >= 2 disq dans les 3 dernières courses → degradation
        if musique:
            import re as _re
            clean_mus = _re.sub(r'\(\d+\)', '', str(musique)).strip()
            tokens_mus = _re.findall(r'[0-9DATRdat][amphsc]', clean_mus)[:3]
            nb_disq_recent = sum(1 for t in tokens_mus if t[0].upper() == 'D')
            if nb_disq_recent >= 2:
                return 'degradation'
        entry = _chrono_cache.get(str(nom).upper().strip())
        if not entry or not isinstance(entry, dict): return 'inconnu'
        hist = entry.get('history', [])
        if len(hist) < 2: return 'stable'
        ecart = hist[0] - hist[-1]
        if ecart < -500:  return 'progres'
        if ecart > 500:   return 'degradation'
        return 'stable'
    df_nc['tendance_chrono'] = df_nc.apply(
        lambda row: _tendance_chrono(row['nom'], row.get('musique', '')), axis=1)

    # ── rang_rk_peloton et ecart_meilleur_rk ────────────────
    # V25 : utiliser reduction_km_v2_ferrure (anti-leakage) au lieu de reduction_km_v2
    # cohérent avec l'entraînement du modèle V25
    rk_vals  = df_nc['reduction_km_v2_ferrure'].values.astype(float)
    rk_clean = np.where((rk_vals > 60000) & (rk_vals < 90000), rk_vals, np.nan)
    rk_med   = float(np.nanmedian(rk_clean)) if np.any(~np.isnan(rk_clean)) else 76000
    rk_fill  = np.where(np.isnan(rk_clean), rk_med, rk_clean)

    n = len(df_nc)
    rk_rank      = pd.Series(rk_fill).rank(ascending=True).values
    rk_rank_norm = 1 - (rk_rank - 1) / max(n - 1, 1)
    df_nc['rang_rk_peloton'] = rk_rank_norm

    # ── ecart_meilleur_rk ─────────────────────────────────────
    rk_min = float(np.nanmin(rk_clean)) if np.any(~np.isnan(rk_clean)) else rk_med
    ecart  = rk_fill - rk_min
    ecart_max = ecart.max()
    if ecart_max > 0:
        df_nc['ecart_meilleur_rk'] = 1 - (ecart / ecart_max)
    else:
        df_nc['ecart_meilleur_rk'] = 0.5

    # ── ratio_gains_peloton — gains vs moyenne peloton ────────
    gains     = df_nc['gains_carriere'].values.astype(float)
    gains_moy = gains.mean()
    if gains_moy > 0:
        df_nc['ratio_gains_peloton'] = (gains / (gains_moy + 1)).clip(0, 5) / 5
    else:
        df_nc['ratio_gains_peloton'] = 0.5

    # ── ratio_victoires_peloton — winrate vs médiane peloton ──
    tv     = df_nc['ratio_victoires'].values.astype(float)
    tv_med = float(np.median(tv))
    tv_std = float(tv.std())
    if tv_std > 1e-6:
        df_nc['ratio_victoires_peloton'] = (
            (tv - tv_med) / (tv_std * 2 + 1e-9)).clip(-1, 1) * 0.5 + 0.5
    else:
        df_nc['ratio_victoires_peloton'] = 0.5

    # ── ratio_age_peloton — age relatif dans le peloton ──────
    age     = df_nc['age'].values.astype(float)
    age_med = float(np.median(age))
    age_std = float(age.std())
    if age_std > 1e-6:
        df_nc['ratio_age_peloton'] = (
            (age - age_med) / (age_std * 2 + 1e-9)).clip(-1, 1) * 0.5 + 0.5
    else:
        df_nc['ratio_age_peloton'] = 0.5

    # ── rang_driver_peloton — rang driver par winrate ─────────
    if 'driver_win_rate_bayes' in df_nc.columns:
        drv_wr = df_nc['driver_win_rate_bayes'].values.astype(float)
        d_rank = pd.Series(drv_wr).rank(ascending=False).values
        df_nc['rang_driver_peloton'] = 1 - (d_rank - 1) / max(n - 1, 1)
    else:
        df_nc['rang_driver_peloton'] = 0.5

    # ════════════════════════════════════════════════════════════
    # ÉTAPE 1 — SCORES MÉTIER (0.0 → 1.0 chacun)
    # Chaque score résume une dimension indépendante.
    # Aucune dimension ne peut écraser les autres.
    # ════════════════════════════════════════════════════════════

    def _norm(series, low, high):
        """Clip + normalise linéairement entre 0 et 1 (bornes absolues)."""
        return ((series.clip(low, high) - low) / (high - low + 1e-9)).clip(0, 1)

    def _norm_rel(series):
        """Normalise relativement au peloton : min→0, max→1.
        Si tous les chevaux sont identiques, retourne 0.5 pour tous."""
        mn, mx = series.min(), series.max()
        if mx - mn < 1e-9:
            return pd.Series(0.5, index=series.index)
        return ((series - mn) / (mx - mn)).clip(0, 1)

    def _norm_mix(series, low, high, rel_weight=0.5):
        """Mix normalisation absolue + relative au peloton."""
        abs_norm = _norm(series, low, high)
        rel_norm = _norm_rel(series)
        return (abs_norm * (1 - rel_weight) + rel_norm * rel_weight).clip(0, 1)

    # ── Score 1 : Forme / Musique ─────────────────────────────
    s_score_p   = _norm_mix(df_nc['mus_score_pondere'],  0, 9)
    s_derniere  = _norm_mix(15 - df_nc['mus_derniere_place'], 0, 14)
    s_podiums   = _norm_mix(df_nc['mus_nb_podiums'],     0, 5)
    s_disq      = 1 - _norm(df_nc['mus_taux_disq'], 0, 0.3)
    s_temps_mus = _norm_mix(df_nc['hist_moy_temps'].fillna(df_nc['hist_moy_temps'].median()).fillna(100), 60, 100)
    s_age       = 1 - _norm(df_nc['age'].fillna(5), 3, 12)
    s_deferre   = df_nc['deferre'].fillna(0).astype(float)

    # Bug fix : pénaliser fortement les disqualifications récentes (3 dernières courses)
    def _nb_disq_recent(musique):
        import re as _re
        if not musique: return 0
        clean = _re.sub(r'\(\d+\)', '', str(musique)).strip()
        tokens = _re.findall(r'[0-9DATRdat][amphsc]', clean)[:3]
        return sum(1 for t in tokens if t[0].upper() == 'D')
    df_nc['_nb_disq_recent'] = df_nc['musique'].apply(_nb_disq_recent)
    # Pénalité : 0 disq recent → 1.0, 1 disq → 0.5, 2+ disq → 0.0
    s_disq_recent = (2 - df_nc['_nb_disq_recent'].clip(0, 2)) / 2

    # ── score_absence — jours depuis la dernière course ──────────────────
    # Source : chrono_cache['date_derniere'] — date réelle dernière course
    # Fallback : 30j si inconnu
    today = pd.Timestamp.now().normalize()
    def _score_absence(row):
        nom = str(row.get('nom','')).upper().strip()
        entry = _chrono_cache.get(nom)
        nb_j = 30  # fallback
        if entry and isinstance(entry, dict):
            date_str = entry.get('date_derniere','')
            if date_str:
                try:
                    date_last = pd.Timestamp(date_str)
                    nb_j = (today - date_last).days
                except: pass
        nb_j = max(0, min(400, nb_j))
        if nb_j <= 7:   return 0.944 + (1.000 - 0.944) * (nb_j / 7)
        if nb_j <= 21:  return 0.944 + (1.000 - 0.944) * ((nb_j-7) / 14)
        if nb_j <= 45:  return 1.000 - (1.000 - 0.918) * ((nb_j-21) / 24)
        if nb_j <= 75:  return 0.918 - (0.918 - 0.762) * ((nb_j-45) / 30)
        if nb_j <= 135: return 0.762 - (0.762 - 0.673) * ((nb_j-75) / 60)
        if nb_j <= 270: return 0.673 - (0.673 - 0.647) * ((nb_j-135) / 135)
        return          0.647 - (0.647 - 0.587) * ((nb_j-270) / 130)
    df_nc['score_absence'] = df_nc.apply(_score_absence, axis=1)

    df_nc['score_forme'] = (
        s_score_p     * 0.25 +
        s_derniere    * 0.20 +
        s_podiums     * 0.12 +
        s_disq        * 0.08 +
        s_disq_recent * 0.20 +  # pénalité disq récentes fortement pondérée
        s_temps_mus   * 0.10 +
        s_age         * 0.03 +
        s_deferre     * 0.02
    ).clip(0, 1)
    df_nc.drop(columns=['_nb_disq_recent'], inplace=True, errors='ignore')

    # ── Score 2 : Duo cheval/driver ───────────────────────────
    s_winrate = _norm_mix(df_nc['duo_win_rate_bayes'], _fallback * 0.8, 0.65)
    s_fiable  = df_nc['duo_fiable'].astype(float)
    s_duo_n   = _norm_mix(df_nc['duo_n'], 1, 15)

    df_nc['score_duo'] = (
        s_winrate * 0.60 +
        s_fiable  * 0.25 +
        s_duo_n   * 0.15
    ).clip(0, 1)

    # ── Score 3 : Historique cheval ───────────────────────────
    hist_taux   = df_nc['hist_taux_top3'].fillna(_fallback)
    hist_class  = df_nc['hist_moy_classement'].fillna(8)
    hist_fiable = _norm(df_nc['hist_nb'], 0, 20)
    hist_tend   = df_nc['hist_tendance'].fillna(0)
    hist_cote   = df_nc['hist_moy_cote'].fillna(df_nc['hist_moy_cote'].median()).fillna(15)

    s_taux_top3  = _norm_mix(hist_taux,        0, 0.7)
    s_classement = _norm_mix(10 - hist_class, -5, 9)
    s_h_fiable   = hist_fiable
    s_tendance   = _norm_mix(hist_tend, -3, 3)                    # tendance positive = en forme
    s_hist_cote  = 1 - _norm(hist_cote, 2, 30)                   # cote historique basse = bon cheval

    df_nc['score_historique'] = (
        s_taux_top3  * 0.35 +
        s_classement * 0.25 +
        s_h_fiable   * 0.10 +
        s_tendance   * 0.20 +
        s_hist_cote  * 0.10
    ).clip(0, 1)

    # ── Score 4 : Gains / Palmarès carrière ───────────────────
    s_ratio_vic   = _norm_mix(df_nc['ratio_victoires'],  0, 0.4)
    s_gains_c     = _norm_mix(df_nc['gains_par_course'], 0, 8000)
    s_gains_ann   = _norm_mix(df_nc['gains_annee'],      0, 150000)
    s_ratio_gains = _norm_mix(df_nc['ratio_gains_rec'],  0, 0.5)
    s_ratio_pl    = _norm_mix(df_nc['ratio_places'],     0, 0.6)  # régularité dans le peloton

    df_nc['score_gains'] = (
        s_ratio_vic   * 0.30 +
        s_gains_c     * 0.25 +
        s_gains_ann   * 0.20 +
        s_ratio_gains * 0.15 +
        s_ratio_pl    * 0.10
    ).clip(0, 1)

    # ── Score 5 : Spécialisation / Adéquation course ─────────
    s_spec_dist  = _norm_mix(df_nc['spec_dist_rate'],       _fallback * 0.8, 0.65)
    s_spec_disc  = _norm_mix(df_nc['spec_disc_rate'],       _fallback * 0.8, 0.65)
    s_entr       = _norm_mix(df_nc['entr_win_rate_bayes'],  _fallback * 0.8, 0.55)
    s_avis       = _norm(df_nc['avis_entraineur'].astype(float), -1, 1)

    df_nc['score_adequation'] = (
        s_spec_dist * 0.35 +
        s_spec_disc * 0.25 +
        s_entr      * 0.25 +
        s_avis      * 0.15
    ).clip(0, 1)

    # ── Score 6 : Cote & marché ───────────────────────────────
    s_cote_rang   = 1 - df_nc['rang_cote_norme']                        # rang inversé : favori = 1
    s_ecart       = _norm(-df_nc['ecart_cotes'].abs(), -10, 0)          # petite déviation live/ref = bon signe
    _med_temps    = df_nc['temps_norme'].median()
    _med_temps    = _med_temps if pd.notna(_med_temps) else 0.0
    s_temps       = _norm(1 / (1 + df_nc['temps_norme'].fillna(_med_temps)), 0, 1)
    s_cote_direct = 1 - _norm_mix(df_nc['rapport_direct'].fillna(df_nc['rapport_ref']), 2, 50)  # cote live basse = favori

    df_nc['score_cote'] = (
        s_cote_rang   * 0.40 +
        s_cote_direct * 0.30 +
        s_ecart       * 0.15 +
        s_temps       * 0.15
    ).clip(0, 1).fillna(s_cote_rang.clip(0, 1))

    # ════════════════════════════════════════════════════════════
    # SCORING FINAL
    #
    # Priorité :
    #   1. XGBoost V12 Ranking → predict() retourne des scores
    #      calibrés en probas par IsotonicRegression
    #   2. XGBoost Classification (V9-V11) → predict_proba()
    #   3. Fallback V6 → somme pondérée fixe
    # ════════════════════════════════════════════════════════════

    SCORES_6 = ['score_forme', 'score_duo', 'score_historique',
                'score_gains', 'score_adequation', 'score_cote']
    POIDS_V6 = [0.21, 0.17, 0.14, 0.08, 0.26, 0.11]

    # Somme pondérée V6 (fallback)
    score_metier = sum(df_nc[s] * p for s, p in zip(SCORES_6, POIDS_V6))
    df_nc['score_metier'] = score_metier

    if _use_v7 and _model_v7 is not None:
        try:
            features_modele = _bundle_v7.get('features', SCORES_6)
            model_type      = _bundle_v7.get('model_type', 'classification')

            df_input = pd.DataFrame(index=df_nc.index)
            for feat in features_modele:
                if feat in df_nc.columns:
                    df_input[feat] = df_nc[feat]
                else:
                    print(f"⚠️  Feature '{feat}' absente — remplacée par 0.5")
                    df_input[feat] = 0.5

            if model_type == 'ranking':
                # XGBRanker.predict() retourne des scores (pas des probas)
                scores_bruts = _model_v7.predict(df_input[features_modele])
                score_brut   = pd.Series(scores_bruts, index=df_nc.index)

                # RANKING : convertir les scores BRUTS en notes
                # Le calibrateur est utilisé uniquement pour proba_pmu (affichage)
                # mais PAS pour les notes — sinon il écrase les écarts
                df_nc['note_pmu'] = _proba_to_note_v7(score_brut)

                # ── Indice de confiance (point 3) ─────────────
                # Plage des scores bruts = dispersion du peloton
                plage_scores = float(score_brut.max() - score_brut.min())
                seuils       = _confiance_seuils
                if plage_scores < seuils.get('faible', 1.24):
                    confiance_course = 'faible'
                elif plage_scores > seuils.get('fort', 1.651):
                    confiance_course = 'fort'
                else:
                    confiance_course = 'moyen'
                df_nc['_plage_scores']    = plage_scores
                df_nc['_confiance']       = confiance_course

                # ── Score value (point 4) ──────────────────────
                # Détecte les outsiders sous-estimés par le marché
                # score_value = note_pmu * log(cote)
                # Valeur élevée = bonne note ET grande cote
                # Favori cote 1.5 note 18 → 18 * log(1.5) = 7.3  (faible value)
                # Outsider cote 25 note 14 → 14 * log(25) = 45.4 (forte value)
                if '_cote_app' in df_nc.columns:
                    cote_val = df_nc['_cote_app'].fillna(10.0)
                    cote_val = cote_val.clip(1.1, 200)
                    df_nc['score_value'] = (
                        df_nc['note_pmu'] * np.log(cote_val)
                    ).round(1)
                else:
                    df_nc['score_value'] = 0.0

                # Calibrateur pour proba_pmu uniquement (affichage %)
                if _calibrator_v9 is not None:
                    try:
                        score_final = pd.Series(
                            _calibrator_v9.predict(score_brut.values),
                            index=df_nc.index)
                    except Exception:
                        score_final = score_brut
                else:
                    score_final = score_brut

                version_utilisee = _bundle_v7.get('version', 'v15')
            else:
                # XGBClassifier.predict_proba()
                probas     = _model_v7.predict_proba(df_input[features_modele])[:, 1]
                poids_cote = _bundle_v7.get('poids_cote_fixe', 0.0)
                poids_xgb  = _bundle_v7.get('poids_xgb', 1.0)
                if poids_cote > 0:
                    score_brut = pd.Series(
                        poids_xgb * probas + poids_cote * df_nc['score_cote'].values,
                        index=df_nc.index)
                else:
                    score_brut = pd.Series(probas, index=df_nc.index)

            # Calibrateur isotonique → scores → probas calibrées
            # (uniquement pour classification — le ranking gère ça ci-dessus)
            if model_type != 'ranking':
                if _calibrator_v9 is not None:
                    try:
                        score_final = pd.Series(
                            _calibrator_v9.predict(score_brut.values),
                            index=df_nc.index)
                    except Exception as e_cal:
                        print(f"⚠️  Calibrateur échoué ({e_cal}) — scores bruts")
                        score_final = score_brut
                else:
                    score_final = score_brut

                # Conversion en notes
                p_min = _bundle_v7.get('proba_min')
                p_max = _bundle_v7.get('proba_max')
                df_nc['note_pmu'] = _proba_to_note_v7(score_final,
                                                       proba_min_ref=p_min,
                                                       proba_max_ref=p_max)

            version_utilisee = _bundle_v7.get('version', 'v12')

        except Exception as e:
            print(f"⚠️  XGBoost predict_proba échoué ({e}) — fallback V6")
            score_final      = score_metier
            df_nc['note_pmu'] = _proba_to_note_api(score_final)
            version_utilisee = "v6_fallback"
    else:
        # V6 : somme pondérée fixe
        score_final      = score_metier
        df_nc['note_pmu'] = _proba_to_note_api(score_final)
        version_utilisee = "v6"

    df_nc['proba_pmu'] = score_final


    # ── Résultat JSON (scores détaillés inclus) ───────────────
    result = []
    for _, row in df_nc.sort_values('note_pmu', ascending=False).iterrows():
        result.append({
            "numero":    int(row['numero']),
            "nom":       str(row['nom']),
            "note_pmu":  int(row['note_pmu']),
            "proba_pmu": round(float(row['proba_pmu']) * 100, 1) if pd.notna(row['proba_pmu']) else 0,
            "driver":    str(row['driver']),
            "cote":      float(row['_cote_app']) if pd.notna(row['_cote_app']) else None,
            "avis":      int(row['avis_entraineur']) if pd.notna(row['avis_entraineur']) else 0,
            # ✨ V6 — scores détaillés par dimension (0-100)
            "scores": {
                "forme":      int(round(float(row['score_forme'])      * 100)) if pd.notna(row['score_forme'])      else 0,
                "duo":        int(round(float(row['score_duo'])        * 100)) if pd.notna(row['score_duo'])        else 0,
                "historique": int(round(float(row['score_historique']) * 100)) if pd.notna(row['score_historique']) else 0,
                "gains":      int(round(float(row['score_gains'])      * 100)) if pd.notna(row['score_gains'])      else 0,
                "adequation": int(round(float(row['score_adequation']) * 100)) if pd.notna(row['score_adequation']) else 0,
                "cote":       int(round(float(row['score_cote'])       * 100)) if pd.notna(row['score_cote'])       else 0,
            },
            "taux_disq":    round(float(row['mus_taux_disq']) * 100, 1) if pd.notna(row.get('mus_taux_disq')) else 0,
            "musique":      str(row.get('musique', '')) if row.get('musique') else '',
            "courses_60j":  int(row['courses_60j']) if pd.notna(row.get('courses_60j')) else 0,
            "score_value":  round(float(row.get('score_value', 0)), 2),
            "nb_courses_base": int(row.get('mus_nb_courses', 0)),
            "rk_brut":         float(row['rk_brut']) if pd.notna(row.get('rk_brut')) and row.get('rk_brut') else None,
            "flag_chrono":     str(row.get('flag_chrono', 'ok')),
            "deferre":         str(row.get('cat_ferrure', 'FERRE')),
            "rk_ferrure":      float(row['reduction_km_v2_ferrure']) if pd.notna(row.get('reduction_km_v2_ferrure')) and row.get('reduction_km_v2_ferrure') else None,
            "tendance_chrono": str(row.get('tendance_chrono', 'inconnu')),
        })

    # Indice de confiance de la course (calculé sur le peloton)
    confiance_course = str(df_nc.get('_confiance', pd.Series(['moyen'])).iloc[0]) \
                       if '_confiance' in df_nc.columns else 'moyen'
    plage_scores     = round(float(df_nc.get('_plage_scores', pd.Series([0])).iloc[0]), 3) \
                       if '_plage_scores' in df_nc.columns else 0

    return jsonify({
        "date":      date_str,
        "reunion":   r_num,
        "course":    c_num,
        "version":   version_utilisee,
        "chevaux":   result,
        "confiance": confiance_course,   # 'faible' / 'moyen' / 'fort'
        "plage":     plage_scores,       # écart max-min des scores bruts
    })


# ============================================================
# TÉLÉCHARGEMENT CSV
# ============================================================
@app.route('/download_historique', methods=['GET'])
def download_historique():
    from flask import send_file
    if os.path.exists(HISTORIQUE_PATH):
        return send_file(
            HISTORIQUE_PATH,
            mimetype='text/csv',
            as_attachment=True,
            download_name='historique_notes.csv'
        )
    else:
        return jsonify({"error": "Fichier non trouvé"}), 404

# ============================================================
# DEBUG — liste des features V5
# ============================================================
@app.route('/features', methods=['GET'])
def get_features():
    if _features_pmu is None:
        return jsonify({"error": "Modèle non chargé"}), 503
    return jsonify({
        "nb_features": len(_features_pmu),
        "features": list(_features_pmu)
    })

# ============================================================
# DEBUG — infos stockage
# ============================================================
@app.route('/storage_info', methods=['GET'])
def storage_info():
    """Indique quel backend de stockage est actif pour /ajouter."""
    conn = _get_pg()
    if conn:
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM courses_manuelles")
                nb = cur.fetchone()[0]
            return jsonify({"stockage": "postgresql", "nb_lignes_manuelles": nb, "ephemere": False})
        except Exception as e:
            return jsonify({"stockage": "postgresql_erreur", "detail": str(e)})
    nb_csv = 0
    if os.path.exists(CSV_PATH):
        try:
            nb_csv = len(pd.read_csv(CSV_PATH))
        except Exception:
            pass
    return jsonify({
        "stockage": "csv_local",
        "ephemere": True,
        "avertissement": "Les données /ajouter seront perdues au redémarrage. Ajoutez DATABASE_URL pour activer PostgreSQL.",
        "nb_lignes_csv": nb_csv,
    })

# ============================================================
# ENTRAÎNEMENT XGBoost V7 — sur les 6 scores métier
# ============================================================
def _calculer_scores_historique(df_hist):
    """
    Applique le même pipeline de scoring V6 sur le DataFrame historique
    (qui contient rang_arrivee) pour produire les features d'entraînement V7.
    Retourne un DataFrame avec les 6 scores + la cible top3.
    """
    d = df_hist.dropna(subset=['rang_arrivee', 'note', 'rapport']).copy()
    if len(d) < 200:
        return None

    # ── Features de base disponibles dans historique_notes.csv ──
    # note, rapport, rang_arrivee sont toujours présents.
    # Les features PMU détaillées (musique, gains, driver…) ne sont PAS
    # dans l'historique CSV — on calcule des scores simplifiés mais cohérents.

    def _norm(s, lo, hi):
        return ((s.clip(lo, hi) - lo) / (hi - lo + 1e-9)).clip(0, 1)

    def _norm_rel(s):
        mn, mx = s.min(), s.max()
        if mx - mn < 1e-9:
            return pd.Series(0.5, index=s.index)
        return ((s - mn) / (mn)) .clip(0, 1) if False else ((s - mn) / (mx - mn)).clip(0, 1)

    def _norm_mix(s, lo, hi):
        return (_norm(s, lo, hi) * 0.5 + _norm_rel(s) * 0.5).clip(0, 1)

    # Score forme — basé sur note (proxy musique)
    d['score_forme']      = _norm_mix(d['note'], 0, 20)

    # Score duo — non disponible dans l'historique CSV → valeur neutre 0.5
    d['score_duo']        = 0.5

    # Score historique — taux top3 par cheval calculé sur l'historique glissant
    grp = d.groupby('nom')['rang_arrivee'].apply(lambda x: (x <= 3).mean()).rename('hist_top3')
    d = d.join(grp, on='nom')
    d['score_historique'] = _norm_mix(d['hist_top3'].fillna(0.3), 0, 0.7)

    # Score gains — non disponible → valeur neutre
    d['score_gains']      = 0.5

    # Score adéquation — non disponible → valeur neutre
    d['score_adequation'] = 0.5

    # Score cote — inversement proportionnel au rapport (favori = score élevé)
    d['score_cote']       = _norm_mix(1.0 / (1.0 + d['rapport']), 0, 0.5)

    d['target_top3'] = (d['rang_arrivee'] <= 3).astype(int)

    SCORES = ['score_forme', 'score_duo', 'score_historique',
              'score_gains', 'score_adequation', 'score_cote']
    return d[SCORES + ['target_top3', 'date']].dropna()


def _entrainer_v7():
    """
    Charge le modèle V10 attelé depuis model_pmu_v10_attele.pkl.
    Contient le modèle XGBoost + calibrateur + snapshots pour l'inférence.
    Pas de réentraînement au démarrage — uniquement chargement.
    """
    global _model_v7, _bundle_v7, _use_v7
    global _calibrator_v9
    global _duo_momentum_snap, _top3_3courses_snap, _top3_60j_snap
    global _fraicheur_snap, _progression_snap, _aptitude_snap
    global _niveau_snap, _confiance_seuils
    global _fallback_rk_v9, _duo_fiable_seuil_v9
    global _driver_stats, _entr_stats, _duo_stats, _spec_dist
    global _prior_pmu, _k_bayes_pmu

    if not os.path.exists(PMU_V7_PATH):
        print(f"⚠️  {PMU_V7_PATH} introuvable — modèle attelé V12 désactivé")
        _use_v7 = False
        return

    try:
        with open(PMU_V7_PATH, 'rb') as f:
            bundle = pickle.load(f)

        _model_v7  = bundle['model']
        _bundle_v7 = bundle
        _use_v7    = True

        # Calibrateur isotonique
        _calibrator_v9 = bundle.get('calibrator')
        if _calibrator_v9 is not None:
            print(f"  ✅ Calibrateur isotonique chargé")
        else:
            print(f"  ⚠️  Pas de calibrateur dans le pkl")

        # Type de modèle (ranking vs classification)
        model_type = bundle.get('model_type', 'classification')
        print(f"  ✅ Type modèle : {model_type}")

        # Snapshots V9
        _duo_momentum_snap  = bundle.get('duo_momentum_snap')
        _top3_3courses_snap = bundle.get('top3_3courses_snap')
        _top3_60j_snap      = bundle.get('top3_60j_snap')

        # Snapshots V10
        _fraicheur_snap   = bundle.get('fraicheur_snap')
        _progression_snap = bundle.get('progression_snap')
        _aptitude_snap    = bundle.get('aptitude_snap')

        # Snapshots V12
        _niveau_snap = bundle.get('niveau_snap')

        # Seuils indice de confiance (V15)
        if bundle.get('confiance_seuils'):
            _confiance_seuils = bundle['confiance_seuils']
            print(f"  ✅ Seuils confiance : "
                  f"faible<{_confiance_seuils['faible']} · "
                  f"fort>{_confiance_seuils['fort']}")

        # Stats
        if bundle.get('driver_stats') is not None:
            _driver_stats = bundle['driver_stats']
        if bundle.get('entr_stats') is not None:
            _entr_stats   = bundle['entr_stats']
        if bundle.get('duo_stats') is not None:
            _duo_stats    = bundle['duo_stats']
        if bundle.get('spec_dist') is not None:
            _spec_dist    = bundle['spec_dist']

        _fallback_rk_v9      = bundle.get('fallback_rk',
                                           {'court': 76000, 'moyen': 75100,
                                            'long': 76000, 'tres_long': 76500})
        _duo_fiable_seuil_v9 = bundle.get('duo_fiable_seuil', 5)
        if bundle.get('prior_win') is not None and _prior_pmu is None:
            _prior_pmu   = bundle['prior_win']
            _k_bayes_pmu = bundle.get('k_bayes', 10)

        # ── Bornes de probabilité pour la conversion en notes ──
        # Si le pkl ne les contient pas (V9 Colab), on les estime
        # depuis prior_win : le modèle produit des probas dans
        # [prior/3, prior*2.5] environ pour un XGBoost calibré
        if 'proba_min' not in bundle:
            prior = bundle.get('prior_win', 0.309)
            bundle['proba_min'] = round(prior * 0.25, 3)   # ≈ 0.077
            bundle['proba_max'] = round(min(prior * 2.5, 0.90), 3)  # ≈ 0.773
            print(f"ℹ️  proba_min/max estimés : [{bundle['proba_min']}, {bundle['proba_max']}]"
                  f" (prior={prior:.3f})")

        ver  = bundle.get('version', 'v9')
        auc  = bundle.get('auc_val', '?')
        nf   = len(bundle.get('features', []))
        n_dm = len(_duo_momentum_snap)  if _duo_momentum_snap  is not None else 0
        n_t3 = len(_top3_3courses_snap) if _top3_3courses_snap is not None else 0
        n_60 = len(_top3_60j_snap)      if _top3_60j_snap      is not None else 0
        print(f"✅ XGBoost {ver} chargé (AUC:{auc} · {nf} features · "
              f"duo_momentum:{n_dm} · top3_3c:{n_t3} · top3_60j:{n_60})")

    except Exception as e:
        print(f"❌ Erreur chargement {PMU_V7_PATH} : {e}")
        _use_v7 = False


# ============================================================
# STATS JOCKEY GALOP
# ============================================================
def _charger_jockey_stats_galop():
    """Charge les stats jockey depuis les pkl PLAT et HAIE."""
    global _jockey_stats_galop
    # Les stats sont embarquées dans model_pmu_plat.pkl
    for pkl_path in [GALOP_MODEL_PATHS.get('PLAT'), GALOP_MODEL_PATHS.get('HAIE')]:
        if not pkl_path or not os.path.exists(pkl_path):
            continue
        try:
            with open(pkl_path, 'rb') as f:
                bundle = pickle.load(f)
            if 'jockey_stats' in bundle:
                _jockey_stats_galop = bundle['jockey_stats']
                n = len(_jockey_stats_galop)
                mx = _jockey_stats_galop['driver_win_rate_bayes'].max()
                print(f"✅ Stats jockey galop chargées depuis {pkl_path} : {n} jockeys (max={mx:.3f})")
                return
        except Exception as e:
            print(f"⚠️  Erreur lecture jockey_stats depuis {pkl_path} : {e}")
    print("⚠️  jockey_stats non trouvées dans les pkl — score_jockey au fallback")


# ============================================================
# CHARGEMENT MODÈLES GALOP
# ============================================================
PLAT_SNAPSHOTS_PATH   = "plat_snapshots.json.gz"
ATTELE_SNAPSHOTS_PATH = "attele_snapshots.json.gz"
# Cache chrono : nom → meilleur rk des 3 dernières courses valides
# Chargé depuis attele_snapshots.json.gz (généré en Colab)
_chrono_cache = {}
_chrono_cache_ferrure = {}          # ATTELÉ nom||cat_ferrure → meilleur rk par ferrure
_chrono_cache_completion = {}       # ATTELÉ nom||cat_ferrure → taux_completion
_chrono_cache_completion_global = {}# ATTELÉ nom → taux_completion toutes ferrures
_chrono_cache_monte = {}            # MONTE nom → meilleur rk
_chrono_cache_ferrure_monte = {}    # MONTE nom||cat_ferrure → meilleur rk par ferrure
_chrono_cache_completion_monte = {} # MONTE nom||cat_ferrure → taux_completion
_chrono_cache_completion_global_monte = {} # MONTE nom → taux_completion toutes ferrures
MONTE_SNAPSHOTS_PATH  = "monte_snapshots.json.gz"
HAIE_SNAPSHOTS_PATH   = "haie_snapshots.json.gz"

def _notes_pmu_haie_v1(df_nc, date_str, r_num, c_num):
    """Pipeline inférence HAIE V1 — XGBoost Ranking."""
    bundle   = _models_galop['HAIE']
    model    = bundle['model']
    feats    = bundle['features']
    cal      = bundle.get('calibrator')
    prior    = bundle.get('prior_win', 0.353)
    k_bayes  = bundle.get('k_bayes', 10)
    fallback = prior * k_bayes / (k_bayes + 1)
    n        = len(df_nc)

    # Features simples
    df_nc['ratio_victoires']  = df_nc['nb_victoires']  / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places']     = df_nc['nb_places']      / (df_nc['nb_courses'] + 1)
    df_nc['gains_par_course'] = df_nc['gains_carriere'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_gains_rec']  = df_nc['gains_annee']    / (df_nc['gains_carriere'] + 1)
    df_nc['nb_partants_c']    = n
    df_nc['tranche_distance'] = pd.cut(df_nc['distance'],
        bins=[0,3000,4000,9999],
        labels=['court','moyen','long']).astype(str)

    # Features peloton
    gains = df_nc['gains_carriere'].values.astype(float)
    gains_moy = gains.mean()
    df_nc['ratio_gains_peloton'] = (
        (gains/(gains_moy+1)).clip(0,5)/5 if gains_moy>0 else 0.5)

    tv = df_nc['ratio_victoires'].values.astype(float)
    tv_med = float(np.median(tv)); tv_std = float(tv.std())
    df_nc['ratio_victoires_peloton'] = (
        ((tv-tv_med)/(tv_std*2+1e-9)).clip(-1,1)*0.5+0.5 if tv_std>1e-6 else 0.5)

    pw = df_nc['handicap_poids'].values.astype(float)
    pw_std = float(pw.std())
    df_nc['poids_relatif_peloton'] = (
        (-(pw-pw.mean())/(pw_std+1e-9)).clip(-3,3)/3*0.5+0.5 if pw_std>1e-6 else 0.5)

    hv = df_nc['handicap_valeur'].values.astype(float)
    hv_rank = pd.Series(hv).rank(ascending=False).values
    df_nc['rang_handicap_norm'] = 1-(hv_rank-1)/max(n-1,1)

    # Jockey
    if _haie_jockey_stats is not None:
        try:
            df_nc = df_nc.merge(
                _haie_jockey_stats[['driver','jockey_win_rate_bayes',
                                    'jockey_win_rate_30j','jockey_n']],
                on='driver', how='left')
        except Exception: pass
    for col, val in [('jockey_win_rate_bayes',fallback),
                     ('jockey_win_rate_30j',fallback),('jockey_n',0)]:
        if col not in df_nc.columns: df_nc[col] = val
        df_nc[col] = df_nc[col].fillna(val)

    jwr = df_nc['jockey_win_rate_bayes'].values.astype(float)
    j_rank = pd.Series(jwr).rank(ascending=False).values
    df_nc['rang_jockey_peloton'] = 1-(j_rank-1)/max(n-1,1)

    # Entraîneur
    if _haie_entr_stats is not None:
        try:
            df_nc = df_nc.merge(
                _haie_entr_stats[['entraineur','entr_win_rate_bayes','entr_win_rate_30j']],
                on='entraineur', how='left')
        except Exception: pass
    for col, val in [('entr_win_rate_bayes',fallback),('entr_win_rate_30j',fallback)]:
        if col not in df_nc.columns: df_nc[col] = val
        df_nc[col] = df_nc[col].fillna(val)

    # Forme récente
    for snap, col, default in [
        (_haie_top3_3c_snap,'top3_3courses',prior),
        (_haie_top3_60j_snap,'top3_60j',prior),
        (_haie_regularite_snap,'regularite_top3',0.0),
    ]:
        if snap is not None and col in snap.columns:
            try: df_nc = df_nc.merge(snap[['nom',col]], on='nom', how='left')
            except Exception: pass
        if col not in df_nc.columns: df_nc[col] = default
        df_nc[col] = df_nc[col].fillna(default)

    # Aptitude distance
    if _haie_apt_dist_snap is not None:
        try:
            df_nc = df_nc.merge(
                _haie_apt_dist_snap[['nom','tranche_distance','apt_dist_recente']],
                on=['nom','tranche_distance'], how='left')
        except Exception: pass
    if 'apt_dist_recente' not in df_nc.columns: df_nc['apt_dist_recente'] = prior
    df_nc['apt_dist_recente'] = df_nc['apt_dist_recente'].fillna(prior)

    # Niveau lot
    if _haie_niveau_lot_snap is not None:
        try:
            df_nc = df_nc.merge(
                _haie_niveau_lot_snap[['nom','niveau_lot_recent']], on='nom', how='left')
        except Exception: pass
    if 'niveau_lot_recent' not in df_nc.columns:
        df_nc['niveau_lot_recent'] = df_nc['montant_prix']
    df_nc['niveau_lot_recent'] = df_nc['niveau_lot_recent'].fillna(df_nc['montant_prix'])
    df_nc['ratio_niveau_lot'] = (df_nc['montant_prix']/(df_nc['niveau_lot_recent']+1)).clip(0,5)
    df_nc['descente_lot']     = (df_nc['ratio_niveau_lot']<0.8).astype(float)

    if _haie_niveau_snap is not None:
        try:
            df_nc = df_nc.merge(
                _haie_niveau_snap[['nom','niveau_habituel']], on='nom', how='left')
        except Exception: pass
    if 'niveau_habituel' not in df_nc.columns:
        df_nc['niveau_habituel'] = df_nc['montant_prix']
    df_nc['niveau_habituel'] = df_nc['niveau_habituel'].fillna(df_nc['montant_prix'])
    df_nc['ratio_niveau']    = (df_nc['montant_prix']/(df_nc['niveau_habituel']+1)).clip(0,5)

    # Gains
    df_nc['gains_annee'] = df_nc['gains_annee'].fillna(0)

    # Prédiction
    df_input = pd.DataFrame(index=df_nc.index)
    for feat in feats:
        df_input[feat] = df_nc[feat] if feat in df_nc.columns else fallback

    scores_bruts = model.predict(df_input[feats])
    score_brut   = pd.Series(scores_bruts, index=df_nc.index)
    df_nc['note_pmu'] = _proba_to_note_v7(score_brut)

    plage_scores = float(score_brut.max()-score_brut.min())
    seuils = _haie_confiance_seuils
    confiance_course = ('fort' if plage_scores>seuils.get('fort',0.7)
                        else 'faible' if plage_scores<seuils.get('faible',0.3)
                        else 'moyen')

    if cal is not None:
        try: score_final = pd.Series(cal.predict(score_brut.values), index=df_nc.index)
        except Exception: score_final = score_brut
    else:
        score_final = score_brut
    df_nc['proba_pmu'] = score_final

    if '_cote_app' in df_nc.columns:
        cote_val = df_nc['_cote_app'].fillna(10.0).clip(1.1,200)
        df_nc['score_value'] = (df_nc['note_pmu']*np.log(cote_val)).round(1)
    else:
        df_nc['score_value'] = 0.0

    # Scores affichage
    df_nc['score_forme']     = df_nc['mus_score_pondere'].fillna(0)/9
    df_nc['score_jockey']    = df_nc['rang_jockey_peloton'].fillna(0.5)
    df_nc['score_duo']       = df_nc['entr_win_rate_bayes'].fillna(prior)
    df_nc['score_historique']= df_nc['top3_3courses'].fillna(prior)
    df_nc['score_gains']     = df_nc['gains_par_course'].fillna(0).clip(0,5000)/5000
    df_nc['score_handicap']  = df_nc['rang_handicap_norm'].fillna(0.5)
    df_nc['score_niveau']    = (1-df_nc['ratio_niveau_lot'].fillna(1).clip(0,2)/2)
    df_nc['score_cote']      = 0.5

    result = []
    for _, row in df_nc.sort_values('note_pmu', ascending=False).iterrows():
        result.append({
            "numero":    int(row['numero']),
            "nom":       str(row['nom']),
            "note_pmu":  int(row['note_pmu']),
            "proba_pmu": round(float(row['proba_pmu'])*100,1) if pd.notna(row['proba_pmu']) else 0,
            "driver":    str(row['driver']),
            "cote":      float(row['_cote_app']) if pd.notna(row.get('_cote_app')) else None,
            "avis":      0,
            "scores": {
                "forme":      int(round(float(row['score_forme'])*100)),
                "duo":        int(round(float(row['score_duo'])*100)),
                "jockey":     int(round(float(row['score_jockey'])*100)),
                "historique": int(round(float(row['score_historique'])*100)),
                "gains":      int(round(float(row['score_gains'])*100)),
                "handicap":   int(round(float(row['score_handicap'])*100)),
                "niveau":     int(round(float(row['score_niveau'])*100)),
                "cote":       0,
            },
            "taux_disq":  round(float(row['mus_taux_disq'])*100,1) if pd.notna(row.get('mus_taux_disq')) else 0,
            "musique":    str(row.get('musique','')) if row.get('musique') else '',
            "score_value": round(float(row.get('score_value',0)),1),
            "handicap_poids":  int(row.get('handicap_poids',0)),
            "handicap_valeur": float(row.get('handicap_valeur',0)),
            "nb_courses_base": int(row.get('mus_nb_courses', 0)),
            "rk_brut":         float(row['rk_brut']) if pd.notna(row.get('rk_brut')) and row.get('rk_brut') else None,
            "flag_chrono":     str(row.get('flag_chrono', 'ok')),
            "deferre":         str(row.get('cat_ferrure', 'FERRE')),
            "rk_ferrure":      float(row['reduction_km_v2_ferrure']) if pd.notna(row.get('reduction_km_v2_ferrure')) and row.get('reduction_km_v2_ferrure') else None,
            "tendance_chrono": str(row.get('tendance_chrono', 'inconnu')),
        })

    return jsonify({
        "date":      date_str,
        "reunion":   r_num,
        "course":    c_num,
        "discipline":"HAIE",
        "version":   bundle.get('version','haie_v1_ranking'),
        "chevaux":   result,
        "confiance": confiance_course,
        "plage":     round(plage_scores,3),
    })


def _charger_snapshots_haie():
    """Charge les snapshots HAIE depuis haie_snapshots.json.gz."""
    global _haie_jockey_stats, _haie_entr_stats
    global _haie_top3_3c_snap, _haie_top3_60j_snap
    global _haie_regularite_snap, _haie_apt_dist_snap
    global _haie_niveau_lot_snap, _haie_niveau_snap, _haie_confiance_seuils

    if not os.path.exists(HAIE_SNAPSHOTS_PATH):
        print(f"⚠️  {HAIE_SNAPSHOTS_PATH} introuvable — snapshots HAIE depuis pkl uniquement")
        return
    try:
        import gzip, json
        print(f"📊 Chargement snapshots HAIE depuis {HAIE_SNAPSHOTS_PATH}…")
        with gzip.open(HAIE_SNAPSHOTS_PATH, 'rt', encoding='utf-8') as f:
            snaps = json.load(f)

        def to_df(key):
            data = snaps.get(key, [])
            return pd.DataFrame(data) if data else None

        _haie_jockey_stats    = to_df('jockey_stats')
        _haie_entr_stats      = to_df('entr_stats')
        _haie_top3_3c_snap    = to_df('top3_3courses_snap')
        _haie_top3_60j_snap   = to_df('top3_60j_snap')
        _haie_regularite_snap = to_df('regularite_snap')
        _haie_apt_dist_snap   = to_df('apt_dist_snap')
        _haie_niveau_lot_snap = to_df('niveau_lot_snap')
        _haie_niveau_snap     = to_df('niveau_snap')

        date_ref = snaps.get('_date_ref','?')
        n_jky = len(_haie_jockey_stats) if _haie_jockey_stats is not None else 0
        n_chx = len(_haie_top3_3c_snap) if _haie_top3_3c_snap is not None else 0
        print(f"✅ Snapshots HAIE chargés depuis JSON (date_ref={date_ref})")
        print(f"   {n_jky} jockeys · {n_chx} chevaux")
    except Exception as e:
        print(f"❌ Erreur chargement snapshots HAIE : {e}")


def _notes_pmu_monte_v1(df_nc, date_str, r_num, c_num):
    """Pipeline inférence MONTE V1 — XGBoost Ranking."""
    bundle  = _models_galop['MONTE']
    model   = bundle['model']
    feats   = bundle['features']
    cal     = bundle.get('calibrator')
    prior   = bundle.get('prior_win', 0.366)
    k_bayes = bundle.get('k_bayes', 10)
    fallback = prior * k_bayes / (k_bayes + 1)
    n       = len(df_nc)

    # Features simples
    df_nc['ratio_victoires']  = df_nc['nb_victoires']  / (df_nc['nb_courses'] + 1)
    df_nc['ratio_places']     = df_nc['nb_places']      / (df_nc['nb_courses'] + 1)
    df_nc['gains_par_course'] = df_nc['gains_carriere'] / (df_nc['nb_courses'] + 1)
    df_nc['ratio_gains_rec']  = df_nc['gains_annee']    / (df_nc['gains_carriere'] + 1)
    df_nc['log_montant_prix'] = np.log1p(df_nc['montant_prix'])
    df_nc['nb_partants_c']    = n
    df_nc['tranche_distance'] = pd.cut(df_nc['distance'],
        bins=[0,1600,2100,2700,9999],
        labels=['court','moyen','long','tres_long']).astype(str)

    # Features peloton
    gains = df_nc['gains_carriere'].values.astype(float)
    gains_moy = gains.mean()
    df_nc['ratio_gains_peloton'] = (
        (gains/(gains_moy+1)).clip(0,5)/5 if gains_moy>0 else 0.5)

    tv = df_nc['ratio_victoires'].values.astype(float)
    tv_med = float(np.median(tv)); tv_std = float(tv.std())
    df_nc['ratio_victoires_peloton'] = (
        ((tv-tv_med)/(tv_std*2+1e-9)).clip(-1,1)*0.5+0.5 if tv_std>1e-6 else 0.5)

    age = df_nc['age'].values.astype(float)
    age_med = float(np.median(age)); age_std = float(age.std())
    df_nc['ratio_age_peloton'] = (
        ((age-age_med)/(age_std*2+1e-9)).clip(-1,1)*0.5+0.5 if age_std>1e-6 else 0.5)

    pw = df_nc['handicap_poids'].values.astype(float)
    pw_std = float(pw.std())
    df_nc['poids_relatif_peloton'] = (
        (-(pw-pw.mean())/(pw_std+1e-9)).clip(-3,3)/3*0.5+0.5 if pw_std>1e-6 else 0.5)

    # Jockey stats
    if _monte_jockey_stats is not None:
        try:
            df_nc = df_nc.merge(
                _monte_jockey_stats[['driver','jockey_win_rate_bayes',
                                     'jockey_win_rate_30j','jockey_n']],
                on='driver', how='left')
        except Exception: pass
    for col, val in [('jockey_win_rate_bayes',fallback),
                     ('jockey_win_rate_30j',fallback),('jockey_n',0)]:
        if col not in df_nc.columns: df_nc[col] = val
        df_nc[col] = df_nc[col].fillna(val)

    jwr = df_nc['jockey_win_rate_bayes'].values.astype(float)
    j_rank = pd.Series(jwr).rank(ascending=False).values
    df_nc['rang_jockey_peloton'] = 1-(j_rank-1)/max(n-1,1)

    # Duo stats
    if _monte_duo_stats is not None:
        try:
            df_nc = df_nc.merge(
                _monte_duo_stats[['nom','driver','duo_win_rate_bayes']],
                on=['nom','driver'], how='left')
        except Exception: pass
    if 'duo_win_rate_bayes' not in df_nc.columns: df_nc['duo_win_rate_bayes'] = fallback
    df_nc['duo_win_rate_bayes'] = df_nc['duo_win_rate_bayes'].fillna(fallback)

    # Entraîneur
    if _monte_entr_stats is not None:
        try:
            df_nc = df_nc.merge(
                _monte_entr_stats[['entraineur','entr_win_rate_bayes','entr_win_rate_30j']],
                on='entraineur', how='left')
        except Exception: pass
    for col, val in [('entr_win_rate_bayes',fallback),('entr_win_rate_30j',fallback)]:
        if col not in df_nc.columns: df_nc[col] = val
        df_nc[col] = df_nc[col].fillna(val)

    # Forme récente
    for snap, col, default in [
        (_monte_top3_3c_snap,'top3_3courses',prior),
        (_monte_top3_60j_snap,'top3_60j',prior),
        (_monte_regularite_snap,'regularite_top3',0.0),
    ]:
        if snap is not None and col in snap.columns:
            try: df_nc = df_nc.merge(snap[['nom',col]], on='nom', how='left')
            except Exception: pass
        if col not in df_nc.columns: df_nc[col] = default
        df_nc[col] = df_nc[col].fillna(default)

    # Aptitude distance
    if _monte_apt_dist_snap is not None:
        try:
            df_nc = df_nc.merge(
                _monte_apt_dist_snap[['nom','tranche_distance','apt_dist_recente']],
                on=['nom','tranche_distance'], how='left')
        except Exception: pass
    if 'apt_dist_recente' not in df_nc.columns: df_nc['apt_dist_recente'] = prior
    df_nc['apt_dist_recente'] = df_nc['apt_dist_recente'].fillna(prior)

    # Niveau lot
    if _monte_niveau_lot_snap is not None:
        try:
            df_nc = df_nc.merge(
                _monte_niveau_lot_snap[['nom','niveau_lot_recent']], on='nom', how='left')
        except Exception: pass
    if 'niveau_lot_recent' not in df_nc.columns:
        df_nc['niveau_lot_recent'] = df_nc['montant_prix']
    df_nc['niveau_lot_recent'] = df_nc['niveau_lot_recent'].fillna(df_nc['montant_prix'])
    df_nc['ratio_niveau_lot'] = (df_nc['montant_prix']/(df_nc['niveau_lot_recent']+1)).clip(0,5)
    df_nc['descente_lot']     = (df_nc['ratio_niveau_lot']<0.8).astype(float)

    if _monte_niveau_snap is not None:
        try:
            df_nc = df_nc.merge(
                _monte_niveau_snap[['nom','niveau_habituel']], on='nom', how='left')
        except Exception: pass
    if 'niveau_habituel' not in df_nc.columns:
        df_nc['niveau_habituel'] = df_nc['montant_prix']
    df_nc['niveau_habituel'] = df_nc['niveau_habituel'].fillna(df_nc['montant_prix'])
    df_nc['ratio_niveau']    = (df_nc['montant_prix']/(df_nc['niveau_habituel']+1)).clip(0,5)

    # ── score_absence — jours depuis la dernière course MONTE ──────────
    today_m = pd.Timestamp.now().normalize()
    def _score_absence_monte(row):
        nom = str(row.get('nom','')).upper().strip()
        entry = _chrono_cache_monte.get(nom)
        nb_j = 30
        if entry and isinstance(entry, dict):
            date_str = entry.get('date_derniere','')
            if date_str:
                try:
                    date_last = pd.Timestamp(date_str)
                    nb_j = (today_m - date_last).days
                except: pass
        nb_j = max(0, min(400, nb_j))
        if nb_j <= 7:   return 0.944 + (1.000 - 0.944) * (nb_j / 7)
        if nb_j <= 21:  return 0.944 + (1.000 - 0.944) * ((nb_j-7) / 14)
        if nb_j <= 45:  return 1.000 - (1.000 - 0.918) * ((nb_j-21) / 24)
        if nb_j <= 75:  return 0.918 - (0.918 - 0.762) * ((nb_j-45) / 30)
        if nb_j <= 135: return 0.762 - (0.762 - 0.673) * ((nb_j-75) / 60)
        if nb_j <= 270: return 0.673 - (0.673 - 0.647) * ((nb_j-135) / 135)
        return          0.647 - (0.647 - 0.587) * ((nb_j-270) / 130)
    df_nc['score_absence'] = df_nc.apply(_score_absence_monte, axis=1)

    # ── cat_ferrure + reduction_km_v2_ferrure ────────────────
    def _get_cat_ferrure_monte(f):
        if isinstance(f, (int, float)):
            fi = int(f)
            if fi == 3: return 'DEFERRE_TOTAL'
            if fi in [1,2]: return 'DEFERRE_PARTIEL'
            return 'FERRE'
        f = str(f)
        if 'ANTERIEURS_POSTERIEURS' in f and 'PROTEGE' not in f: return 'DEFERRE_TOTAL'
        if 'DEFERRE' in f: return 'DEFERRE_PARTIEL'
        return 'FERRE'

    if 'deferre' in df_nc.columns:
        df_nc['cat_ferrure'] = df_nc['deferre'].apply(_get_cat_ferrure_monte)
    else:
        df_nc['cat_ferrure'] = 'FERRE'

    def _get_rk_ferrure_monte(row):
        nom = str(row.get('nom','')).upper().strip()
        ferrure = row.get('cat_ferrure','FERRE')
        key = f"{nom}||{ferrure}"
        entry = _chrono_cache_ferrure_monte.get(key)
        if entry and isinstance(entry, dict):
            hist = entry.get('history', [])
            valides = [r for r in hist if 60000 < r < 90000]
            if valides:
                n = len(valides)
                poids = [n - i for i in range(n)]
                total = sum(poids)
                return sum(v * p for v, p in zip(valides, poids)) / total
        # Fallback sur cache global
        entry2 = _chrono_cache_monte.get(nom)
        if entry2 and isinstance(entry2, dict):
            hist2 = entry2.get('history', [])
            valides2 = [r for r in hist2 if 60000 < r < 90000]
            if valides2:
                n2 = len(valides2)
                poids2 = [n2 - i for i in range(n2)]
                total2 = sum(poids2)
                return sum(v * p for v, p in zip(valides2, poids2)) / total2
        return 76000.0

    df_nc['reduction_km_v2_ferrure'] = df_nc.apply(_get_rk_ferrure_monte, axis=1)

    # ── taux_completion_ferrure — fiabilité chrono par ferrure MONTE ─────
    # Fallback : taux ferrure → taux global toutes ferrures → prior
    def _get_completion_ferrure_monte(row):
        nom = str(row.get('nom','')).upper().strip()
        ferrure = row.get('cat_ferrure','FERRE')
        key = f"{nom}||{ferrure}"
        # 1 — Taux par ferrure spécifique
        entry = _chrono_cache_ferrure_monte.get(key)
        if entry and isinstance(entry, dict):
            hist = entry.get('history', [])
            if len(hist) > 0:
                return sum(1 for r in hist if 60000 < r < 90000) / len(hist)
        # 2 — Fallback : taux toutes ferrures
        entry_g = _chrono_cache_monte.get(nom)
        if entry_g and isinstance(entry_g, dict):
            hist_g = entry_g.get('history', [])
            if len(hist_g) > 0:
                return sum(1 for r in hist_g if 60000 < r < 90000) / len(hist_g)
        return 0.77  # prior global MONTE (~77% courses avec chrono valide)
    df_nc['taux_completion_ferrure'] = df_nc.apply(_get_completion_ferrure_monte, axis=1)

    # Prédiction
    df_input = pd.DataFrame(index=df_nc.index)
    for feat in feats:
        df_input[feat] = df_nc[feat] if feat in df_nc.columns else fallback

    scores_bruts = model.predict(df_input[feats])
    score_brut   = pd.Series(scores_bruts, index=df_nc.index)
    df_nc['note_pmu'] = _proba_to_note_v7(score_brut)

    plage_scores = float(score_brut.max()-score_brut.min())
    seuils = _monte_confiance_seuils
    confiance_course = ('fort' if plage_scores>seuils.get('fort',0.7)
                        else 'faible' if plage_scores<seuils.get('faible',0.3)
                        else 'moyen')

    if cal is not None:
        try: score_final = pd.Series(cal.predict(score_brut.values), index=df_nc.index)
        except Exception: score_final = score_brut
    else:
        score_final = score_brut
    df_nc['proba_pmu'] = score_final

    if '_cote_app' in df_nc.columns:
        cote_val = df_nc['_cote_app'].fillna(10.0).clip(1.1,200)
        df_nc['score_value'] = (df_nc['note_pmu']*np.log(cote_val)).round(1)
    else:
        df_nc['score_value'] = 0.0

    df_nc['score_forme']     = df_nc['mus_score_pondere'].fillna(0)/9
    df_nc['score_jockey']    = df_nc['rang_jockey_peloton'].fillna(0.5)
    df_nc['score_duo']       = df_nc['duo_win_rate_bayes'].fillna(prior)
    df_nc['score_historique']= df_nc['top3_3courses'].fillna(prior)
    df_nc['score_gains']     = df_nc['ratio_victoires'].fillna(0).clip(0,0.5)*2
    df_nc['score_handicap']  = df_nc['ratio_victoires_peloton'].fillna(0.5)
    df_nc['score_niveau']    = (1-df_nc['ratio_niveau_lot'].fillna(1).clip(0,2)/2)
    df_nc['score_cote']      = 0.5

    result = []
    for _, row in df_nc.sort_values('note_pmu', ascending=False).iterrows():
        result.append({
            "numero":    int(row['numero']),
            "nom":       str(row['nom']),
            "note_pmu":  int(row['note_pmu']),
            "proba_pmu": round(float(row['proba_pmu'])*100,1) if pd.notna(row['proba_pmu']) else 0,
            "driver":    str(row['driver']),
            "cote":      float(row['_cote_app']) if pd.notna(row.get('_cote_app')) else None,
            "avis":      int(row['avis_entraineur']) if pd.notna(row.get('avis_entraineur')) else 0,
            "scores": {
                "forme":      int(round(float(row['score_forme'])*100)),
                "duo":        int(round(float(row['score_duo'])*100)),
                "jockey":     int(round(float(row['score_jockey'])*100)),
                "historique": int(round(float(row['score_historique'])*100)),
                "gains":      int(round(float(row['score_gains'])*100)),
                "handicap":   int(round(float(row['score_handicap'])*100)),
                "niveau":     int(round(float(row['score_niveau'])*100)),
                "cote":       0,
            },
            "taux_disq":  round(float(row['mus_taux_disq'])*100,1) if pd.notna(row.get('mus_taux_disq')) else 0,
            "musique":    str(row.get('musique','')) if row.get('musique') else '',
            "score_value": round(float(row.get('score_value',0)),1),
            "handicap_poids":  int(row.get('handicap_poids',0)),
            "handicap_valeur": float(row.get('handicap_valeur',0)),
            "nb_courses_base": int(row.get('mus_nb_courses', 0)),
            "rk_brut":         float(row['rk_brut']) if pd.notna(row.get('rk_brut')) and row.get('rk_brut') else None,
            "flag_chrono":     str(row.get('flag_chrono', 'ok')),
            "deferre":         str(row.get('cat_ferrure', 'FERRE')),
            "rk_ferrure":      float(row['reduction_km_v2_ferrure']) if pd.notna(row.get('reduction_km_v2_ferrure')) and row.get('reduction_km_v2_ferrure') else None,
            "tendance_chrono": str(row.get('tendance_chrono', 'inconnu')),
        })

    return jsonify({
        "date":      date_str,
        "reunion":   r_num,
        "course":    c_num,
        "discipline":"MONTE",
        "version":   bundle.get('version','monte_v1_ranking'),
        "chevaux":   result,
        "confiance": confiance_course,
        "plage":     round(plage_scores,3),
    })


def _charger_snapshots_monte():
    """Charge les snapshots MONTE depuis monte_snapshots.json.gz."""
    global _monte_jockey_stats, _monte_duo_stats, _monte_entr_stats
    global _monte_top3_3c_snap, _monte_top3_60j_snap
    global _monte_regularite_snap, _monte_apt_dist_snap
    global _monte_niveau_lot_snap, _monte_niveau_snap, _monte_confiance_seuils

    if not os.path.exists(MONTE_SNAPSHOTS_PATH):
        print(f"⚠️  {MONTE_SNAPSHOTS_PATH} introuvable — snapshots MONTE depuis pkl uniquement")
        return
    try:
        import gzip, json
        print(f"📊 Chargement snapshots MONTE depuis {MONTE_SNAPSHOTS_PATH}…")
        with gzip.open(MONTE_SNAPSHOTS_PATH, 'rt', encoding='utf-8') as f:
            snaps = json.load(f)

        def to_df(key):
            data = snaps.get(key, [])
            return pd.DataFrame(data) if data else None

        _monte_jockey_stats    = to_df('jockey_stats')
        _monte_duo_stats       = to_df('duo_stats')
        _monte_entr_stats      = to_df('entr_stats')
        _monte_top3_3c_snap    = to_df('top3_3courses_snap')
        _monte_top3_60j_snap   = to_df('top3_60j_snap')
        _monte_regularite_snap = to_df('regularite_snap')
        _monte_apt_dist_snap   = to_df('apt_dist_snap')
        _monte_niveau_lot_snap = to_df('niveau_lot_snap')
        _monte_niveau_snap     = to_df('niveau_snap')

        # Charger chrono_cache_monte et chrono_cache_ferrure_monte
        global _chrono_cache_monte, _chrono_cache_ferrure_monte
        for key, v in snaps.get('chrono_cache_monte', {}).items():
            _chrono_cache_monte[key] = v if isinstance(v, dict) else {'min':float(v),'last':float(v),'history':[float(v)]}
        for key, v in snaps.get('chrono_cache_ferrure_monte', {}).items():
            _chrono_cache_ferrure_monte[key] = v if isinstance(v, dict) else {'min':float(v),'last':float(v),'history':[float(v)]}

        date_ref = snaps.get('_date_ref','?')
        n_jky = len(_monte_jockey_stats) if _monte_jockey_stats is not None else 0
        n_chx = len(_monte_top3_3c_snap) if _monte_top3_3c_snap is not None else 0
        print(f"✅ Snapshots MONTE chargés depuis JSON (date_ref={date_ref})")
        print(f"   {n_jky} jockeys · {n_chx} chevaux")
        print(f"   {len(_chrono_cache_monte)} chevaux dans chrono_cache_monte")
        print(f"   {len(_chrono_cache_ferrure_monte)} entrées dans chrono_cache_ferrure_monte")
    except Exception as e:
        print(f"❌ Erreur chargement snapshots MONTE : {e}")


def _charger_snapshots_attele():
    """
    Charge les snapshots attelé depuis attele_snapshots.json.gz.
    Écrase les snapshots figés du pkl V15 avec des données à jour.
    À regénérer lors de chaque réentraînement (toutes les 2 semaines).
    """
    global _driver_stats, _entr_stats, _duo_stats
    global _duo_momentum_snap, _top3_3courses_snap, _top3_60j_snap
    global _niveau_snap

    if not os.path.exists(ATTELE_SNAPSHOTS_PATH):
        print(f"⚠️  {ATTELE_SNAPSHOTS_PATH} introuvable — snapshots attelé depuis pkl uniquement")
        return

    try:
        import gzip, json
        print(f"📊 Chargement snapshots attelé depuis {ATTELE_SNAPSHOTS_PATH}…")
        with gzip.open(ATTELE_SNAPSHOTS_PATH, 'rt', encoding='utf-8') as f:
            snaps = json.load(f)

        def to_df(key):
            data = snaps.get(key, [])
            return pd.DataFrame(data) if data else None

        _driver_stats         = to_df('driver_stats')
        _duo_stats            = to_df('duo_stats')
        _entr_stats           = to_df('entr_stats')
        _duo_momentum_snap    = to_df('duo_momentum_snap')
        _top3_3courses_snap   = to_df('top3_3courses_snap')
        _top3_60j_snap        = to_df('top3_60j_snap')
        _niveau_snap          = to_df('niveau_snap')

        # Chrono cache — format {min, last, history} ou float (ancien format)
        chrono_raw = snaps.get('chrono_cache', {})
        for k, v in chrono_raw.items():
            key = k.upper().strip()
            if isinstance(v, dict):
                _chrono_cache[key] = v
            else:
                # Ancien format float → convertir
                _chrono_cache[key] = {'min': float(v), 'last': float(v), 'history': [float(v)]}

        # Charger chrono_cache_ferrure
        global _chrono_cache_ferrure
        chrono_ferrure_raw = snaps.get('chrono_cache_ferrure', {})
        for key, v in chrono_ferrure_raw.items():
            if isinstance(v, dict):
                _chrono_cache_ferrure[key] = v
            else:
                _chrono_cache_ferrure[key] = {'min': float(v), 'last': float(v), 'history': [float(v)]}

        date_ref = snaps.get('_date_ref', '?')
        n_drv    = len(_driver_stats)   if _driver_stats   is not None else 0
        n_entr   = len(_entr_stats)     if _entr_stats     is not None else 0
        n_chx    = len(_top3_3courses_snap) if _top3_3courses_snap is not None else 0
        print(f"✅ Snapshots attelé chargés depuis JSON (date_ref={date_ref})")
        print(f"   {n_drv} drivers · {n_entr} entraîneurs · {n_chx} chevaux")
        print(f"   {len(_chrono_cache)} chevaux dans chrono_cache")
        print(f"   {len(_chrono_cache_ferrure)} entrées dans chrono_cache_ferrure")

    except Exception as e:
        print(f"❌ Erreur chargement snapshots attelé : {e}")
        import traceback; traceback.print_exc()


def _charger_stats_plat():
    """
    Charge les snapshots PLAT depuis plat_snapshots.json.gz (2.7 Mo).
    Écrase les snapshots figés du pkl avec des données à jour.
    À regénérer lors de chaque réentraînement.
    """
    global _plat_jockey_stats, _plat_duo_stats, _plat_entr_stats
    global _plat_top3_3c_snap, _plat_top3_60j_snap, _plat_regularite_snap
    global _plat_aptitude_terrain, _plat_apt_dist_snap
    global _plat_dernier_jockey_snap, _plat_apt_type_piste_snap
    global _plat_apt_terrain_label_snap
    global _plat_aptitude_hippo_snap, _plat_jockey_hippo_stats
    global _plat_niveau_lot_snap, _plat_niveau_snap

    if not os.path.exists(PLAT_SNAPSHOTS_PATH):
        print(f"⚠️  {PLAT_SNAPSHOTS_PATH} introuvable — snapshots PLAT depuis pkl uniquement")
        return

    try:
        import gzip, json
        print(f"📊 Chargement snapshots PLAT depuis {PLAT_SNAPSHOTS_PATH}…")
        with gzip.open(PLAT_SNAPSHOTS_PATH, 'rt', encoding='utf-8') as f:
            snaps = json.load(f)

        def to_df(key):
            data = snaps.get(key, [])
            return pd.DataFrame(data) if data else None

        _plat_jockey_stats        = to_df('jockey_stats')
        _plat_jockey_hippo_stats  = to_df('jockey_hippo_stats')
        _plat_duo_stats           = to_df('duo_stats')
        _plat_entr_stats          = to_df('entr_stats')
        _plat_top3_3c_snap        = to_df('top3_3courses_snap')
        _plat_top3_60j_snap       = to_df('top3_60j_snap')
        _plat_regularite_snap     = to_df('regularite_snap')
        _plat_apt_dist_snap       = to_df('apt_dist_snap')
        _plat_aptitude_hippo_snap = to_df('aptitude_hippo_snap')
        _plat_niveau_lot_snap     = to_df('niveau_lot_snap')
        _plat_niveau_snap         = to_df('niveau_snap')

        n_jky  = len(_plat_jockey_stats)  if _plat_jockey_stats  is not None else 0
        n_entr = len(_plat_entr_stats)    if _plat_entr_stats    is not None else 0
        n_chx  = len(_plat_top3_3c_snap)  if _plat_top3_3c_snap  is not None else 0
        n_hip  = len(_plat_jockey_hippo_stats) if _plat_jockey_hippo_stats is not None else 0
        print(f"✅ Snapshots PLAT chargés depuis JSON")
        print(f"   {n_jky} jockeys · {n_entr} entraîneurs · {n_chx} chevaux · {n_hip} jockey×hippo")

    except Exception as e:
        print(f"❌ Erreur chargement snapshots PLAT : {e}")
        import traceback; traceback.print_exc()


def _charger_modeles_galop():
    """Charge les modèles XGBoost galop depuis les pkl."""
    global _models_galop
    global _plat_jockey_stats, _plat_duo_stats, _plat_entr_stats
    global _plat_top3_3c_snap, _plat_top3_60j_snap
    global _plat_aptitude_terrain, _plat_aptitude_distance
    global _plat_apt_dist_snap, _plat_regularite_snap, _plat_niveau_lot_snap
    global _plat_niveau_snap, _plat_jockey_hippo_stats, _plat_aptitude_hippo_snap
    global _plat_confiance_seuils

    for disc, path in GALOP_MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"⚠️  {path} introuvable — {disc} désactivé")
            continue
        try:
            with open(path, 'rb') as f:
                bundle = pickle.load(f)
            _models_galop[disc] = bundle
            version  = bundle.get('version', '?')
            auc      = bundle.get('auc_val', bundle.get('auc_final_val', '?'))
            mod_type = bundle.get('model_type', 'classification')
            print(f"✅ Modèle galop {disc} chargé (v={version} · AUC={auc} · {mod_type})")

            # Charger les snapshots PLAT V1 si c'est le nouveau bundle ranking
            if disc == 'PLAT' and mod_type == 'ranking':
                _plat_jockey_stats      = bundle.get('jockey_stats')
                _plat_duo_stats         = bundle.get('duo_stats')
                _plat_entr_stats        = bundle.get('entr_stats')
                _plat_top3_3c_snap      = bundle.get('top3_3courses_snap')
                _plat_top3_60j_snap     = bundle.get('top3_60j_snap')
                _plat_aptitude_terrain  = bundle.get('apt_terrain_snap') or bundle.get('aptitude_terrain_snap')
                _plat_apt_type_piste_snap = bundle.get('apt_type_piste_snap')
                _plat_apt_terrain_label_snap = bundle.get('apt_terrain_label_snap')
                _plat_aptitude_distance = bundle.get('aptitude_distance_snap')
                _plat_apt_dist_snap     = bundle.get('apt_dist_snap')
                _plat_regularite_snap   = bundle.get('regularite_snap')
                _plat_niveau_lot_snap   = bundle.get('niveau_lot_snap')
                _plat_niveau_snap       = bundle.get('niveau_snap')
                _plat_jockey_hippo_stats  = bundle.get('jockey_hippo_stats')
                _plat_aptitude_hippo_snap = bundle.get('aptitude_hippo_snap')
                _plat_dernier_jockey_snap = bundle.get('dernier_jockey_snap')
                if bundle.get('confiance_seuils'):
                    _plat_confiance_seuils = bundle['confiance_seuils']
                # Note : ces snapshots seront écrasés par _charger_stats_plat()
                # si historique_galop_plat_enrichi.csv est disponible
                print(f"  ✅ Snapshots PLAT V4 chargés (seront mis à jour si historique disponible)")

            # ── MONTE V1 Ranking ─────────────────────────────
            if disc == 'MONTE' and mod_type == 'ranking':
                global _monte_jockey_stats, _monte_duo_stats, _monte_entr_stats
                global _monte_top3_3c_snap, _monte_top3_60j_snap
                global _monte_regularite_snap, _monte_apt_dist_snap
                global _monte_niveau_lot_snap, _monte_niveau_snap
                global _monte_confiance_seuils
                _monte_jockey_stats     = bundle.get('jockey_stats')
                _monte_duo_stats        = bundle.get('duo_stats')
                _monte_entr_stats       = bundle.get('entr_stats')
                _monte_top3_3c_snap     = bundle.get('top3_3courses_snap')
                _monte_top3_60j_snap    = bundle.get('top3_60j_snap')
                _monte_regularite_snap  = bundle.get('regularite_snap')
                _monte_apt_dist_snap    = bundle.get('apt_dist_snap')
                _monte_niveau_lot_snap  = bundle.get('niveau_lot_snap')
                _monte_niveau_snap      = bundle.get('niveau_snap')
                if bundle.get('confiance_seuils'):
                    _monte_confiance_seuils = bundle['confiance_seuils']
                print(f"  ✅ Snapshots MONTE V1 chargés")

            # ── HAIE V1 Ranking ──────────────────────────────
            if disc == 'HAIE' and mod_type == 'ranking':
                global _haie_jockey_stats, _haie_entr_stats
                global _haie_top3_3c_snap, _haie_top3_60j_snap
                global _haie_regularite_snap, _haie_apt_dist_snap
                global _haie_niveau_lot_snap, _haie_niveau_snap
                global _haie_confiance_seuils
                _haie_jockey_stats    = bundle.get('jockey_stats')
                _haie_entr_stats      = bundle.get('entr_stats')
                _haie_top3_3c_snap    = bundle.get('top3_3courses_snap')
                _haie_top3_60j_snap   = bundle.get('top3_60j_snap')
                _haie_regularite_snap = bundle.get('regularite_snap')
                _haie_apt_dist_snap   = bundle.get('apt_dist_snap')
                _haie_niveau_lot_snap = bundle.get('niveau_lot_snap')
                _haie_niveau_snap     = bundle.get('niveau_snap')
                if bundle.get('confiance_seuils'):
                    _haie_confiance_seuils = bundle['confiance_seuils']
                print(f"  ✅ Snapshots HAIE V1 chargés")

        except Exception as e:
            print(f"❌ Erreur chargement {path} : {e}")
    print(f"✅ {len(_models_galop)} modèles galop chargés")


# ============================================================
# DÉMARRAGE
# ============================================================
_charger_modele_pmu()
initialiser()
_entrainer_v7()
_charger_modeles_galop()
_charger_snapshots_attele()
_charger_snapshots_monte()
_charger_snapshots_haie()
_charger_stats_plat()
_charger_jockey_stats_galop()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
