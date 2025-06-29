from flask import Flask, request, render_template_string
import joblib, re

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("protein_rf_foodname_model.joblib")
vectorizer = joblib.load("protein_foodname_vectorizer.joblib")

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
    if food_name.endswith("s") and len(food_name) > 3:
        return food_name[:-1]
    return food_name

def parse_item(item_str):
    m = re.match(r"([\d\.]+)?\s*([a-z]+)?\s*(.*)", item_str.strip())
    if not m:
        return 1.0, item_str.strip(), 100
    qty_str, unit, food = m.groups()
    qty = float(qty_str) if qty_str else 1.0
    unit = unit if unit else ""
    food = clean_text(food) or clean_text(unit)
    food = singularize_food_name(food)
    weight = qty * unit_weights.get(unit, 100)
    return qty, food, weight

def predict_protein(food_clean):
    X_vec = vectorizer.transform([food_clean])
    return model.predict(X_vec)[0]

@app.route("/", methods=["GET", "POST"])
def index():
    results, total_protein, items = None, None, ""
    if request.method == "POST":
        items = request.form.get("items", "")
        items_raw = [x.strip() for x in items.split(" and ") if x.strip()]
        results, total_protein = [], 0.0

        for item in items_raw:
            qty, food_clean, weight_g = parse_item(item)
            prot_100 = predict_protein(food_clean)
            prot_tot = prot_100 * weight_g / 100
            total_protein += prot_tot
            results.append({
                "input": f"{qty} {food_clean}",
                "matched": food_clean,
                "weight": weight_g,
                "per100": prot_100,
                "total": prot_tot,
            })

    return render_template_string("""
<!DOCTYPE html>
<html>
<head>
    <title>Protein Predictor</title>
    <style>
        body { font-family: system-ui, sans-serif; max-width: 700px; margin: 2rem auto; }
        h1 { text-align: center; }
        form { display: flex; gap: .5rem; margin-bottom: 1rem; }
        input[type=text] { flex: 1; padding: .5rem .7rem; font-size: 1rem; }
        button { padding: .5rem 1rem; font-size: 1rem; cursor: pointer; }
        table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
        th, td { padding: .4rem .6rem; border-bottom: 1px solid #ddd; text-align: left; }
        tfoot td { font-weight: bold; }
    </style>
</head>
<body>
    <h1>🔍 Protein Predictor</h1>
    <p>Examples: <code>2 eggs and 1 cup of milk</code>, <code>3 bananas and 1 slice of bread</code></p>
    <form method="POST">
        <input type="text" name="items" placeholder="Enter food items..." required value="{{ items }}">
        <button type="submit">Estimate 💪</button>
    </form>

    {% if results %}
    <table>
        <thead>
            <tr><th>Input</th><th>Weight (g)</th><th>Protein /100g</th><th>Estimated Protein (g)</th></tr>
        </thead>
        <tbody>
            {% for r in results %}
            <tr>
                <td>{{ r.input }}</td>
                <td>{{ "%.0f"|format(r.weight) }}</td>
                <td>{{ "%.2f"|format(r.per100) }}</td>
                <td>{{ "%.2f"|format(r.total) }}</td>
            </tr>
            {% endfor %}
        </tbody>
        <tfoot>
            <tr>
                <td colspan="3">Total protein</td>
                <td>{{ "%.2f"|format(total_protein) }} g</td>
            </tr>
        </tfoot>
    </table>
    {% endif %}
</body>
</html>
""", results=results, total_protein=total_protein, items=items)

if __name__ == "__main__":
    app.run(debug=True)
