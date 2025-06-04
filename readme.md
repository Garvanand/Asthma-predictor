# Asthma Prediction Model 🫁

Hey there! 👋 This is my project where I made a cool asthma prediction system using AI and stuff. It's working pretty fine

## What's This All About? 🤔

So basically, I made this streamlit based web app that can predict if someone might have asthma by looking at their symptoms and stuff. It's like having a doctor in your pocket (but not really, pls go to a real doctor if you're sick!).

## The Tech Stuff (Don't Worry, It's Not Too Complicated) 💻

### What I Used:
- Python (cuz it's easy to understand)
- Streamlit (makes pretty web pages)
- Scikit-learn (for the AI magic)
- Plotly (makes cool graphs)
- OpenAQ API (for air quality data)

### The AI Part 🤖
I used a Random Forest Classifier (it's like a bunch of decision trees working together) because:
- It's pretty accurate but constantly working to increase its accuracy (current-88.7% accurate!)
- Doesn't overfit

### The Data 📊
I used a dataset with:
- 15,000+ patient records
- 7 different symptoms
- Age groups and gender info
- Real air quality data from OpenAQ

The symptoms we look at are:
- Tiredness
- Dry cough
- Breathing problems
- Sore throat
- Body pains
- Stuffy nose
- Runny nose

## How to Run This Thing 🚀

1. First, install the stuff you need:
```bash
pip install -r requirements.txt
```

2. Make a .env file and put your OpenAQ API key in it:
```
X-API-Key=your_api_key_here
```

3. Run the app:
```bash
streamlit run main.py
```

## The Cool Features 🌟

- Real-time air quality data (thanks OpenAQ!)
- Pretty graphs and charts
- Risk assessment gauge
- Personalized recommendations
- Emergency contact info

## Accuracy and Stuff 📈

Our model is pretty good at predicting:
- Mild Asthma: 88.6% accuracy
- Moderate Asthma: 84.2% accuracy
- No Asthma: 95% accuracy


## Future Improvements (If I Had More Time) 🔮

- Add more symptoms
- Use more advanced AI models
- Add user accounts
- Make it work on phones better
- Add more languages

## Important Note ⚠️

This is just a project for learning! Don't use it to diagnose yourself - go to a real doctor if you're worried about asthma!

## Credits 🙏

- OpenAQ for the air quality data
- Stack Overflow for fixing my bugs
- Also claude ai for helping out in rectifying few errors

## Contact Me 📧

If you find any bugs or have suggestions, pls let me know! I'm still learning and would love feedback!
---