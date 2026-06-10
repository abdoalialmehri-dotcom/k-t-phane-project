book_summary = """
Pride and Prejudice is a novel by Jane Austen.
It tells the story of Elizabeth Bennet and her family.
The novel focuses on love, marriage, social class,
family relationships, reputation, and personal growth.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read book
with open("book.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Split chapters
chapters = text.split("CHAPTER ")[1:]

# Book summary
book_summary = (
    "A novel about love, marriage, social class, "
    "family relationships and personal growth."
)

# TF-IDF
documents = [book_summary] + chapters

vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)

# Cosine Similarity
similarities = cosine_similarity(
    tfidf_matrix[0:1],
    tfidf_matrix[1:]
)

# Book information
book_title = "Pride and Prejudice"

print(f"\nBook: {book_title}")
print(f"Total Chapters: {len(chapters)}")

# Find lowest similarity chapters
scores = []

for i, score in enumerate(similarities[0], start=1):
    scores.append((i, score))

scores.sort(key=lambda x: x[1])

print("\nLowest Similarity Chapters:")

for chapter_num, score in scores[:5]:
    print(f"Chapter {chapter_num} -> {score:.4f}")