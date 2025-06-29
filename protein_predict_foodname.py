import re
import joblib
import numpy as np

# Load model and vectorizer
model = joblib.load("protein_rf_foodname_model.joblib")
vectorizer = joblib.load("protein_foodname_vectorizer.joblib")

# Approximate unit weights in grams
unit_weights = {
    "g": 1, "gram": 1, "grams": 1,
    "kg": 1000, "kilogram": 1000, "kilograms": 1000,
    "cup": 240, "cups": 240,
    "slice": 30, "slices": 30,
    "piece": 50, "pieces": 50,
    "egg": 50, "eggs": 50,
    "tbsp": 15, "tablespoon": 15, "tablespoons": 15,
    "tsp": 5, "teaspoon": 5, "teaspoons": 5,
}

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def singularize_food_name(food_name):
    # Simple heuristic: remove trailing 's' if it exists and word length > 3 to avoid cutting short words wrongly
    if food_name.endswith('s') and len(food_name) > 3:
        return food_name[:-1]
    return food_name

def parse_item(item_str):
    pattern = r"([\d\.]+)?\s*([a-z]+)?\s*(.*)"
    m = re.match(pattern, item_str.strip())
    if not m:
        return 1.0, item_str.strip(), 100

    qty_str, unit, food = m.groups()
    qty = float(qty_str) if qty_str else 1.0
    unit = unit if unit else ""

    food = food.strip()
    food = clean_text(food)
    if not food:
        food = clean_text(unit)

    # Normalize plural to singular
    food = singularize_food_name(food)

    weight = qty * unit_weights.get(unit, 100)
    return qty, food, weight

def predict_protein(food_name_clean):
    X_vec = vectorizer.transform([food_name_clean])
    pred = model.predict(X_vec)[0]
    return pred

print("🔍 Protein Predictor")
print("📝 Enter foods like:")
print("   → '2 eggs and 1 cup of milk'")      
print("   → '3 bananas and 1 slice of bread'\n")

while True:
    user_input = input("Enter food items (or type 'exit'): ").strip()
    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    items = [x.strip() for x in user_input.split(" and ")]
    total_protein = 0.0
    print("\n🔍 Estimating protein content...\n")

    for item in items:
        qty, food_clean, weight_g = parse_item(item)
        protein_per_100g = predict_protein(food_clean)
        protein_total = protein_per_100g * weight_g / 100
        total_protein += protein_total

        print(f"🍽️ Input: {qty} {food_clean}")
        print(f"🔎 Matched food: {food_clean}")
        print(f"📏 Estimated total weight: {weight_g:.0f}g")
        print(f"🔬 Protein per 100g: {protein_per_100g:.2f}g")
        print(f"💪 Estimated protein: **{protein_total:.2f}g**\n")

    print(f"📊 ✅ Total protein for all items: **{total_protein:.2f}g**\n")
