import sqlite3

# Membuat koneksi ke database SQLite
conn = sqlite3.connect('netflix.db')  # Ganti dengan nama file database Anda
cursor = conn.cursor()

# Data film dan genre baru
film_posterurl_updates = [
    ('https://media.themoviedb.org/t/p/w600_and_h900_bestv2/leWNtbo3AsAiLIdQ2j5BNCtdFQ8.jpg', 'Dua Hati Biru'),
]

# Query untuk mengupdate genre film
cursor.executemany("""
UPDATE movies
SET poster_url = ?
WHERE title = ?
""", film_posterurl_updates)

# Menyimpan perubahan
conn.commit()

# Menutup koneksi
conn.close()
