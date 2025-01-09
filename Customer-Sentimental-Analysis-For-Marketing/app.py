import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from textblob import TextBlob

# Function to load data from CSV
def load_data_from_csv(csv_file):
    return pd.read_csv(csv_file)

# Function to perform sentiment analysis
def get_sentiment(review):
    analysis = TextBlob(review)
    return "Positive" if analysis.sentiment.polarity > 0 else "Negative" if analysis.sentiment.polarity < 0 else "Neutral"

# Load the CSV file (You should replace 'customer_reviews.csv' with the actual file name)
df = load_data_from_csv(r"E:\Data Science Mini Projects\Customer Sentimental Analysis For Marketing\CustomerReview.csv")

# Perform sentiment analysis on reviews
df['Sentiment_Analysis'] = df['Review'].apply(get_sentiment)

# Add a 'Review_Length' column for the length of each review
df['Review_Length'] = df['Review'].apply(lambda x: len(x.split()))

# Add a 'Review_Date' column for demo purposes (if you don't have it, you can add manually or simulate)
if 'Review_Date' not in df.columns:
    df['Review_Date'] = pd.to_datetime(pd.Series([
        "2023-11-01", "2023-11-02", "2023-11-03", "2023-11-04", "2023-11-05"
    ]))

# Streamlit layout
st.title('Customer Sentiment Analysis for Marketing')
st.subheader("Analyze Customer Feedback to Improve Marketing Strategies")

# Show the data table
st.write("### Customer Reviews Data")
st.dataframe(df)

# Customer input for new review and rating
st.subheader("Submit Your Review")
new_review = st.text_area("Enter your review:")
new_rating = st.slider("Rate the product (1 to 5)", 1, 5, 3)

if st.button("Submit Review"):
    new_sentiment = get_sentiment(new_review)
    st.write(f"Sentiment of your review: {new_sentiment}")

    # Add the new review to the DataFrame using pd.concat
    new_id = len(df) + 1
    new_review_df = pd.DataFrame([{"Review_ID": new_id, "Customer_ID": new_id, "Review": new_review, 
                                   "Rating": new_rating, "Sentiment": new_sentiment, 
                                   "Sentiment_Analysis": new_sentiment, "Review_Length": len(new_review.split())}])
    df = pd.concat([df, new_review_df], ignore_index=True)

    st.write("### Updated Data")
    st.dataframe(df)

# 1. **Review Length Distribution**
st.subheader("Review Length Distribution")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(df['Review_Length'], kde=True, ax=ax, color="skyblue")
ax.set_title('Distribution of Review Lengths')
ax.set_xlabel('Number of Words in Review')
ax.set_ylabel('Frequency')
st.pyplot(fig)

# 2. **Sentiment Over Time** (Trend of Sentiments by Review Date)
df['Month'] = df['Review_Date'].dt.to_period('M')
sentiment_monthly = df.groupby(['Month', 'Sentiment_Analysis']).size().unstack().fillna(0)

st.subheader("Sentiment Over Time: Understand Customer Mood Trends")
fig, ax = plt.subplots(figsize=(10, 6))
sentiment_monthly.plot(kind='line', ax=ax, marker='o', title="Sentiment Distribution by Month")
st.pyplot(fig)

# 3. **Word Cloud by Sentiment**
st.subheader("Word Cloud: Most Common Words in Reviews")
text = " ".join(review for review in df.Review)
wordcloud = WordCloud(width=800, height=400, background_color="white").generate(text)
fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")
st.pyplot(fig)

# 4. **Word Cloud by Sentiment (Positive/Negative)**
positive_reviews = " ".join(review for review in df[df['Sentiment_Analysis'] == 'Positive']['Review'])
negative_reviews = " ".join(review for review in df[df['Sentiment_Analysis'] == 'Negative']['Review'])

wordcloud_positive = WordCloud(width=800, height=400, background_color="white").generate(positive_reviews)
wordcloud_negative = WordCloud(width=800, height=400, background_color="white").generate(negative_reviews)

fig, ax = plt.subplots(1, 2, figsize=(16, 8))
ax[0].imshow(wordcloud_positive, interpolation="bilinear")
ax[0].axis("off")
ax[0].set_title("Positive Reviews Word Cloud")

ax[1].imshow(wordcloud_negative, interpolation="bilinear")
ax[1].axis("off")
ax[1].set_title("Negative Reviews Word Cloud")

st.subheader("Word Cloud Comparison: Positive vs Negative Reviews")
st.pyplot(fig)

# 5. **Top N Positive and Negative Reviews**
top_n = 5
st.subheader(f"Top {top_n} Positive Reviews Based on Rating")
st.write(df.nlargest(top_n, 'Rating')[['Review', 'Rating', 'Sentiment_Analysis']])

st.subheader(f"Top {top_n} Negative Reviews Based on Rating")
st.write(df.nsmallest(top_n, 'Rating')[['Review', 'Rating', 'Sentiment_Analysis']])

# 6. **Sentiment Distribution (Pie Chart)**
st.subheader("Customer Sentiment Distribution")
sentiment_counts = df['Sentiment_Analysis'].value_counts()
fig, ax = plt.subplots(figsize=(6, 6))
sentiment_counts.plot.pie(autopct='%1.1f%%', startangle=90, colors=["#66b3ff", "#ff6666", "#99ff99"], ax=ax)
ax.set_ylabel('')
st.pyplot(fig)

# 7. **Ratings by Customer**
st.subheader("Customer Ratings Distribution: Understand Customer Satisfaction")
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=df['Customer_ID'], y=df['Rating'], palette="Blues_d", ax=ax)
ax.set_title("Customer Ratings")
ax.set_xlabel("Customer ID")
ax.set_ylabel("Rating")
st.pyplot(fig)

# 8. **Sentiment vs Rating (Scatter Plot)**
st.subheader("Sentiment vs Rating: Analyzing Customer Feedback")
fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=df['Rating'], y=df['Sentiment_Analysis'].map({'Positive': 1, 'Neutral': 0, 'Negative': -1}), ax=ax, hue=df['Sentiment_Analysis'], palette="coolwarm")
ax.set_title("Sentiment vs Rating")
ax.set_xlabel("Rating")
ax.set_ylabel("Sentiment (Encoded)")
st.pyplot(fig)

# 9. **Review Length vs Sentiment**
st.subheader("Review Length vs Sentiment: Correlation between Review Length and Sentiment")
fig, ax = plt.subplots(figsize=(8, 6))
sns.boxplot(x=df['Sentiment_Analysis'], y=df['Review_Length'], ax=ax, palette="Set2")
ax.set_title("Review Length Distribution by Sentiment")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Review Length (Number of Words)")
st.pyplot(fig)
