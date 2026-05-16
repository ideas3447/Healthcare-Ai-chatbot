"""
Dataset Generator — mimics the Kaggle Disease-Symptom dataset structure.
Produces: dataset.csv, symptom_severity.csv, symptom_description.csv
"""

import pandas as pd
import random

# ── 132 symptoms (same vocabulary as Kaggle dataset) ──────────────────────────
SYMPTOMS = [
    "itching","skin_rash","nodal_skin_eruptions","continuous_sneezing","shivering",
    "chills","joint_pain","stomach_pain","acidity","ulcers_on_tongue","muscle_wasting",
    "vomiting","burning_micturition","spotting_urination","fatigue","weight_gain",
    "anxiety","cold_hands_and_feets","mood_swings","weight_loss","restlessness",
    "lethargy","patches_in_throat","irregular_sugar_level","cough","high_fever",
    "sunken_eyes","breathlessness","sweating","dehydration","indigestion","headache",
    "yellowish_skin","dark_urine","nausea","loss_of_appetite","pain_behind_the_eyes",
    "back_pain","constipation","abdominal_pain","diarrhoea","mild_fever","yellow_urine",
    "yellowing_of_eyes","acute_liver_failure","fluid_overload","swelling_of_stomach",
    "swelled_lymph_nodes","malaise","blurred_and_distorted_vision","phlegm",
    "throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion",
    "chest_pain","weakness_in_limbs","fast_heart_rate","pain_during_bowel_movements",
    "pain_in_anal_region","bloody_stool","irritation_in_anus","neck_pain","dizziness",
    "cramps","bruising","obesity","swollen_legs","swollen_blood_vessels",
    "puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremities",
    "excessive_hunger","extra_marital_contacts","drying_and_tingling_lips","slurred_speech",
    "knee_pain","hip_joint_pain","muscle_weakness","stiff_neck","swelling_joints",
    "movement_stiffness","spinning_movements","loss_of_balance","unsteadiness",
    "weakness_of_one_body_side","loss_of_smell","bladder_discomfort","foul_smell_of_urine",
    "continuous_feel_of_urine","passage_of_gases","internal_itching","toxic_look_(typhos)",
    "depression","irritability","muscle_pain","altered_sensorium","red_spots_over_body",
    "belly_pain","abnormal_menstruation","dischromic_patches","watering_from_eyes",
    "increased_appetite","polyuria","family_history","mucoid_sputum","rusty_sputum",
    "lack_of_concentration","visual_disturbances","receiving_blood_transfusion",
    "receiving_unsterile_injections","coma","stomach_bleeding","distention_of_abdomen",
    "history_of_alcohol_consumption","fluid_overload","blood_in_sputum",
    "prominent_veins_on_calf","palpitations","painful_walking","pus_filled_pimples",
    "blackheads","scurring","skin_peeling","silver_like_dusting","small_dents_in_nails",
    "inflammatory_nails","blister","red_sore_around_nose","yellow_crust_ooze"
]

# ── 41 diseases with their symptom sets ───────────────────────────────────────
DISEASE_SYMPTOMS = {
    "Fungal infection":       ["itching","skin_rash","nodal_skin_eruptions","dischromic_patches"],
    "Allergy":                ["continuous_sneezing","shivering","chills","watering_from_eyes"],
    "GERD":                   ["stomach_pain","acidity","ulcers_on_tongue","vomiting","cough","chest_pain"],
    "Chronic cholestasis":    ["itching","vomiting","yellowish_skin","nausea","loss_of_appetite","abdominal_pain"],
    "Drug Reaction":          ["itching","skin_rash","stomach_pain","burning_micturition","fatigue"],
    "Peptic ulcer disease":   ["vomiting","indigestion","loss_of_appetite","abdominal_pain","passage_of_gases","belly_pain"],
    "AIDS":                   ["muscle_wasting","patches_in_throat","high_fever","extra_marital_contacts","fatigue"],
    "Diabetes":               ["fatigue","weight_loss","restlessness","lethargy","irregular_sugar_level","polyuria","increased_appetite","family_history"],
    "Gastroenteritis":        ["vomiting","sunken_eyes","dehydration","diarrhoea"],
    "Bronchial Asthma":       ["fatigue","cough","high_fever","breathlessness","family_history","mucoid_sputum"],
    "Hypertension":           ["headache","chest_pain","dizziness","loss_of_balance","lack_of_concentration"],
    "Migraine":               ["acidity","indigestion","headache","blurred_and_distorted_vision","excessive_hunger","stiff_neck","depression","irritability"],
    "Cervical spondylosis":   ["back_pain","weakness_in_limbs","neck_pain","dizziness","loss_of_balance"],
    "Paralysis (brain hemorrhage)": ["vomiting","headache","weakness_of_one_body_side","altered_sensorium","slurred_speech"],
    "Jaundice":               ["itching","vomiting","fatigue","weight_loss","high_fever","yellowish_skin","dark_urine","abdominal_pain"],
    "Malaria":                ["chills","vomiting","high_fever","sweating","headache","nausea","diarrhoea","muscle_pain"],
    "Chicken pox":            ["itching","skin_rash","fatigue","lethargy","high_fever","headache","loss_of_appetite","mild_fever","swelled_lymph_nodes","malaise","red_spots_over_body","phlegm"],
    "Dengue":                 ["skin_rash","chills","joint_pain","vomiting","fatigue","high_fever","headache","nausea","loss_of_appetite","pain_behind_the_eyes","back_pain","malaise","muscle_pain","red_spots_over_body"],
    "Typhoid":                ["chills","vomiting","fatigue","high_fever","headache","nausea","constipation","abdominal_pain","diarrhoea","toxic_look_(typhos)","belly_pain"],
    "Hepatitis A":            ["joint_pain","vomiting","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","diarrhoea","mild_fever","yellowing_of_eyes","muscle_pain"],
    "Hepatitis B":            ["itching","fatigue","lethargy","yellowish_skin","dark_urine","loss_of_appetite","abdominal_pain","yellow_urine","yellowing_of_eyes","malaise","receiving_blood_transfusion","receiving_unsterile_injections"],
    "Hepatitis C":            ["fatigue","yellowish_skin","nausea","loss_of_appetite","yellowing_of_eyes","family_history"],
    "Hepatitis D":            ["joint_pain","vomiting","fatigue","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","yellowing_of_eyes"],
    "Hepatitis E":            ["joint_pain","vomiting","fatigue","high_fever","yellowish_skin","dark_urine","nausea","loss_of_appetite","abdominal_pain","yellowing_of_eyes","coma","stomach_bleeding"],
    "Alcoholic hepatitis":    ["vomiting","yellowish_skin","abdominal_pain","swelling_of_stomach","history_of_alcohol_consumption","fluid_overload","renal_failure"],
    "Tuberculosis":           ["chills","vomiting","fatigue","weight_loss","cough","high_fever","breathlessness","sweating","loss_of_appetite","mild_fever","swelled_lymph_nodes","malaise","phlegm","blood_in_sputum","rusty_sputum"],
    "Common Cold":            ["continuous_sneezing","chills","fatigue","cough","high_fever","headache","swelled_lymph_nodes","malaise","phlegm","throat_irritation","redness_of_eyes","sinus_pressure","runny_nose","congestion","chest_pain","loss_of_smell","muscle_pain"],
    "Pneumonia":              ["chills","fatigue","cough","high_fever","breathlessness","sweating","malaise","phlegm","chest_pain","fast_heart_rate","rusty_sputum"],
    "Dimorphic hemorrhoids (piles)": ["constipation","pain_during_bowel_movements","pain_in_anal_region","bloody_stool","irritation_in_anus"],
    "Heart attack":           ["vomiting","breathlessness","sweating","chest_pain","fast_heart_rate"],
    "Varicose veins":         ["fatigue","cramps","bruising","obesity","swollen_legs","swollen_blood_vessels","prominent_veins_on_calf"],
    "Hypothyroidism":         ["fatigue","weight_gain","cold_hands_and_feets","mood_swings","lethargy","puffy_face_and_eyes","enlarged_thyroid","brittle_nails","swollen_extremities","depression","irritability","abnormal_menstruation"],
    "Hyperthyroidism":        ["fatigue","mood_swings","weight_loss","restlessness","sweating","diarrhoea","fast_heart_rate","excessive_hunger","muscle_weakness","irritability","abnormal_menstruation"],
    "Hypoglycemia":           ["vomiting","fatigue","anxiety","sweating","headache","nausea","blurred_and_distorted_vision","excessive_hunger","slurred_speech","irritability","drying_and_tingling_lips","palpitations"],
    "Osteoarthritis":         ["joint_pain","neck_pain","knee_pain","hip_joint_pain","swelling_joints","painful_walking"],
    "Arthritis":              ["muscle_weakness","stiff_neck","swelling_joints","movement_stiffness","loss_of_balance"],
    "(Vertigo) Paroxysmal Positional Vertigo": ["vomiting","headache","nausea","spinning_movements","loss_of_balance","unsteadiness"],
    "Acne":                   ["skin_rash","pus_filled_pimples","blackheads","scurring"],
    "Urinary tract infection":["burning_micturition","bladder_discomfort","foul_smell_of_urine","continuous_feel_of_urine"],
    "Psoriasis":              ["skin_rash","joint_pain","skin_peeling","silver_like_dusting","small_dents_in_nails","inflammatory_nails"],
    "Impetigo":               ["skin_rash","high_fever","blister","red_sore_around_nose","yellow_crust_ooze"],
}

# ── Emergency severity scores (0–7 scale, higher = more dangerous) ────────────
SEVERITY = {
    "itching":1,"skin_rash":3,"nodal_skin_eruptions":4,"continuous_sneezing":4,
    "shivering":5,"chills":3,"joint_pain":3,"stomach_pain":5,"acidity":3,
    "ulcers_on_tongue":4,"muscle_wasting":3,"vomiting":5,"burning_micturition":6,
    "spotting_urination":6,"fatigue":4,"weight_gain":3,"anxiety":4,
    "cold_hands_and_feets":5,"mood_swings":3,"weight_loss":4,"restlessness":5,
    "lethargy":2,"patches_in_throat":6,"irregular_sugar_level":5,"cough":4,
    "high_fever":6,"sunken_eyes":5,"breathlessness":7,"sweating":3,
    "dehydration":4,"indigestion":5,"headache":3,"yellowish_skin":3,
    "dark_urine":4,"nausea":5,"loss_of_appetite":4,"pain_behind_the_eyes":4,
    "back_pain":3,"constipation":4,"abdominal_pain":6,"diarrhoea":6,
    "mild_fever":5,"yellow_urine":3,"yellowing_of_eyes":3,"acute_liver_failure":7,
    "fluid_overload":6,"swelling_of_stomach":7,"swelled_lymph_nodes":6,
    "malaise":5,"blurred_and_distorted_vision":5,"phlegm":3,"throat_irritation":4,
    "redness_of_eyes":3,"sinus_pressure":4,"runny_nose":3,"congestion":4,
    "chest_pain":7,"weakness_in_limbs":6,"fast_heart_rate":6,
    "pain_during_bowel_movements":6,"pain_in_anal_region":7,"bloody_stool":7,
    "irritation_in_anus":6,"neck_pain":5,"dizziness":4,"cramps":4,"bruising":4,
    "obesity":3,"swollen_legs":5,"swollen_blood_vessels":5,"puffy_face_and_eyes":3,
    "enlarged_thyroid":3,"brittle_nails":3,"swollen_extremities":5,
    "excessive_hunger":4,"extra_marital_contacts":5,"drying_and_tingling_lips":4,
    "slurred_speech":7,"knee_pain":3,"hip_joint_pain":3,"muscle_weakness":3,
    "stiff_neck":4,"swelling_joints":4,"movement_stiffness":5,"spinning_movements":6,
    "loss_of_balance":5,"unsteadiness":5,"weakness_of_one_body_side":7,
    "loss_of_smell":3,"bladder_discomfort":4,"foul_smell_of_urine":5,
    "continuous_feel_of_urine":6,"passage_of_gases":5,"internal_itching":4,
    "toxic_look_(typhos)":5,"depression":3,"irritability":2,"muscle_pain":2,
    "altered_sensorium":7,"red_spots_over_body":5,"belly_pain":4,
    "abnormal_menstruation":3,"dischromic_patches":2,"watering_from_eyes":4,
    "increased_appetite":5,"polyuria":3,"family_history":7,"mucoid_sputum":4,
    "rusty_sputum":5,"lack_of_concentration":4,"visual_disturbances":5,
    "receiving_blood_transfusion":7,"receiving_unsterile_injections":6,"coma":7,
    "stomach_bleeding":7,"distention_of_abdomen":6,"history_of_alcohol_consumption":5,
    "blood_in_sputum":7,"prominent_veins_on_calf":5,"palpitations":4,
    "painful_walking":2,"pus_filled_pimples":2,"blackheads":1,"scurring":3,
    "skin_peeling":3,"silver_like_dusting":2,"small_dents_in_nails":1,
    "inflammatory_nails":2,"blister":4,"red_sore_around_nose":3,"yellow_crust_ooze":1,
    "renal_failure":7,
}

# ── Disease descriptions ───────────────────────────────────────────────────────
DESCRIPTIONS = {
    "Fungal infection": "A fungal infection is caused by fungi affecting skin, nails, or mucous membranes. Consult a dermatologist for antifungal treatment.",
    "Allergy": "An allergy is an immune reaction to a foreign substance. Avoid triggers and consult an allergist.",
    "GERD": "Gastroesophageal reflux disease causes stomach acid to flow into the esophagus. Dietary changes and medication help.",
    "Chronic cholestasis": "A liver condition where bile flow is impaired. Requires medical evaluation and management.",
    "Drug Reaction": "An adverse reaction to medication. Stop the suspected drug and consult your doctor immediately.",
    "Peptic ulcer disease": "Sores in the stomach lining or small intestine. Caused by H. pylori or NSAIDs. Requires medical treatment.",
    "AIDS": "Advanced stage of HIV infection affecting the immune system. Requires immediate specialist care and antiretroviral therapy.",
    "Diabetes": "A metabolic disorder with elevated blood glucose. Requires lifestyle changes and medication. Monitor blood sugar regularly.",
    "Gastroenteritis": "Inflammation of the stomach and intestines, often from infection. Stay hydrated and rest.",
    "Bronchial Asthma": "A chronic respiratory condition causing airway inflammation. Use inhalers as prescribed and avoid triggers.",
    "Hypertension": "High blood pressure that can lead to serious complications. Requires medication and lifestyle changes.",
    "Migraine": "Severe recurrent headaches with nausea and sensitivity to light. Consult a neurologist for management.",
    "Cervical spondylosis": "Age-related wear on the cervical spine. Physiotherapy and pain management are recommended.",
    "Paralysis (brain hemorrhage)": "Bleeding in the brain causing loss of function. THIS IS A MEDICAL EMERGENCY. Call emergency services immediately.",
    "Jaundice": "Yellowing of skin/eyes due to liver dysfunction. Requires immediate medical evaluation.",
    "Malaria": "A mosquito-borne infection causing fever and chills. Requires antimalarial medication promptly.",
    "Chicken pox": "A viral infection causing itchy rash and blisters. Rest, fluids, and antiviral medication if severe.",
    "Dengue": "A mosquito-borne viral disease with severe fever. Monitor platelet count and seek medical care.",
    "Typhoid": "A bacterial infection from contaminated food/water. Requires antibiotics and medical supervision.",
    "Hepatitis A": "A viral liver infection spread through contaminated food/water. Rest and supportive care required.",
    "Hepatitis B": "A viral liver infection spread through blood/body fluids. Requires antiviral treatment.",
    "Hepatitis C": "A blood-borne viral liver infection. Highly treatable with modern antiviral drugs.",
    "Hepatitis D": "A liver infection that occurs with Hepatitis B. Requires specialist care.",
    "Hepatitis E": "A viral liver disease from contaminated water. Usually self-limiting but can be severe in pregnancy.",
    "Alcoholic hepatitis": "Liver inflammation from excessive alcohol use. Stop alcohol immediately and seek medical care.",
    "Tuberculosis": "A bacterial lung infection spread through air. Requires a long course of antibiotics.",
    "Common Cold": "A mild viral respiratory infection. Rest, fluids, and over-the-counter medication for symptom relief.",
    "Pneumonia": "A lung infection causing inflammation. Requires antibiotics and possibly hospitalization.",
    "Dimorphic hemorrhoids (piles)": "Swollen veins in the rectum or anus. Dietary fiber, hydration, and medical treatment help.",
    "Heart attack": "Blockage of blood flow to the heart. THIS IS A MEDICAL EMERGENCY. Call emergency services immediately.",
    "Varicose veins": "Enlarged, twisted veins usually in the legs. Compression stockings and medical evaluation recommended.",
    "Hypothyroidism": "Underactive thyroid gland. Requires thyroid hormone replacement therapy.",
    "Hyperthyroidism": "Overactive thyroid gland. Requires medication or other treatments to reduce thyroid activity.",
    "Hypoglycemia": "Low blood sugar causing weakness and confusion. Consume glucose immediately and consult your doctor.",
    "Osteoarthritis": "Degeneration of joint cartilage. Pain management, physiotherapy, and lifestyle changes help.",
    "Arthritis": "Inflammation of joints causing pain and stiffness. Medical treatment and physiotherapy are recommended.",
    "(Vertigo) Paroxysmal Positional Vertigo": "Brief episodes of dizziness from inner ear issues. Repositioning maneuvers and medication help.",
    "Acne": "A skin condition causing pimples and blackheads. Topical treatments and dermatologist care recommended.",
    "Urinary tract infection": "Bacterial infection of the urinary tract. Requires antibiotics. Drink plenty of water.",
    "Psoriasis": "A chronic skin condition causing red, scaly patches. Requires dermatologist-prescribed treatment.",
    "Impetigo": "A highly contagious bacterial skin infection. Requires antibiotic treatment.",
}

# ── Emergency diseases (any prediction of these triggers emergency alert) ─────
EMERGENCY_DISEASES = {
    "Heart attack", "Paralysis (brain hemorrhage)", "Hepatitis E",
    "Tuberculosis", "Dengue", "AIDS", "Pneumonia", "Alcoholic hepatitis"
}

def build_dataset(samples_per_disease=50):
    rows = []
    max_syms = 17
    for disease, syms in DISEASE_SYMPTOMS.items():
        for _ in range(samples_per_disease):
            # randomly pick 3–len(syms) symptoms from the disease's set
            k = random.randint(3, len(syms))
            chosen = random.sample(syms, k)
            row = {f"Symptom_{i+1}": (chosen[i] if i < len(chosen) else None)
                   for i in range(max_syms)}
            row["Disease"] = disease
            rows.append(row)
    return pd.DataFrame(rows)

if __name__ == "__main__":
    random.seed(42)
    df = build_dataset(60)
    df.to_csv("dataset.csv", index=False)
    print(f"dataset.csv — {len(df)} rows, {df['Disease'].nunique()} diseases")

    sev_df = pd.DataFrame(
        [(s, w) for s, w in SEVERITY.items()],
        columns=["Symptom", "weight"]
    )
    sev_df.to_csv("symptom_severity.csv", index=False)
    print(f"symptom_severity.csv — {len(sev_df)} symptoms")

    desc_df = pd.DataFrame(
        [(d, t) for d, t in DESCRIPTIONS.items()],
        columns=["Disease", "Description"]
    )
    desc_df.to_csv("symptom_description.csv", index=False)
    print(f"symptom_description.csv — {len(desc_df)} diseases")

    print("\nEmergency diseases:", EMERGENCY_DISEASES)
    print("\nDone.")
