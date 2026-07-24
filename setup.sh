#!/bin/bash
echo "=========================================="
echo "🎓 Graduate Admission Predictor - ANN Setup"
echo "=========================================="

# Create directories
echo "📁 Creating directories..."
mkdir -p data models saved_visualizations

# Install dependencies
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Place 'Admission_Predict_Ver1.1.csv' in 'data/' folder"
echo "2. Train model: python train_model.py"
echo "3. Run app: streamlit run app.py"
echo "=========================================="