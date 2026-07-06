from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    print("=" * 55)
    print("      🤖 PHISHING EMAIL DETECTION MODEL (ML) 🤖      ")
    print("=" * 55)

    # 1. Dataset: Textual content and URL simulation keywords
    # 0 = Safe/Legitimate Email, 1 = Phishing Email
    emails = [
        "Hey, are we still meeting for lunch today at 1 PM?",
        "URGENT: Your bank account is locked. Click http://secure-bank-login.com to verify!",
        "The project report is attached. Please review and give feedback.",
        "Congratulations! You won a $1000 gift card. Claim your prize at http://win-free-money.xyz",
        "Can you send me the update link for the weekly schedule?",
        "Dear customer, suspicious activity detected on your account. Reset password now.",
        "Please find the invoice for your subscription renewal inside.",
        "OFFICIAL: Update your login credentials immediately at http://paypal-security-update.net",
        "Hi Mom, I will visit you this weekend. Let me know if you need anything.",
        "Get rich quick! Click this link to earn $5000 a day guaranteed!"
    ]
    
    # Labels corresponding to the emails above
    labels = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    print("[+] Processing dataset and extracting features...")
    
    # 2. Feature Extraction: Convert text into numerical vectors using TF-IDF
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(emails)
    y = labels

    # 3. Train/Test Split (Using minimal split for simulation dataset)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # 4. Model Training: Using Naive Bayes Classifier (Great for text classification)
    model = MultinomialNB()
    model.fit(X_train, y_train)
    print("[+] Machine Learning model trained successfully.")

    # 5. Model Evaluation
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)

    print("\n" + "-"*25 + " EVALUATION " + "-"*25)
    print(f"📊 Model Accuracy: {accuracy * 100:.2f}%")
    
    # Simple Text-Based Confusion Matrix Display
    print("\n🧩 Confusion Matrix:")
    print(f"   Predicted Safe  Predicted Phishing")
    print(f"Actual Safe       [{cm[0][0]}]               [{cm[0][1]}]")
    print(f"Actual Phishing   [{cm[1][0]}]               [{cm[1][1]}]")
    print("-"*62)

    # 6. Live Testing Interface
    print("\n[+] Test the Model Live:")
    user_email = input("Enter an email body text or link to test: ").strip()
    
    if user_email:
        # Transform user input using the trained vectorizer
        user_features = vectorizer.transform([user_email])
        prediction = model.predict(user_features)[0]
        
        print("\n" + "="*40)
        print(f"Analyzed Text: \"{user_email}\"")
        if prediction == 1:
            print("🚨 RESULT: WARNING! This looks like a PHISHING EMAIL.")
        else:
            print("✅ RESULT: This email appears to be SAFE.")
        print("="*40)

if __name__ == "__main__":
    main()
