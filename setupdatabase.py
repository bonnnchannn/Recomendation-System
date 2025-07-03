import sqlite3

# Membuat koneksi ke database SQLite
conn = sqlite3.connect('netflix.db')  # Ganti dengan nama file database Anda
cursor = conn.cursor()

# Data film yang ingin dimasukkan
film_data = [
    ('Scandal Makers', 'Comedy', 2023, 'Oscar, seorang penyiar radio dan selebriti yang tiba-tiba didatangi oleh anak beserta cucunya. Sebagai seorang penyiar radio dan selebriti terkenal tentunya ia berusaha membuat citranya harum di depan publik.', 'https://upload.wikimedia.org/wikipedia/id/2/23/Poster_film_Scandal_Makers_%282023%29.jpeg'),
]

# Query untuk menambahkan film baru
cursor.executemany("""
INSERT INTO movies (title, genre, year, description, poster_url)
VALUES (?, ?, ?, ?, ?)
""", film_data)  # Menambahkan dua baris data sekaligus


cursor.execute("""
UPDATE movies
SET genre = ?
WHERE title = ?
""", ('Romance, Comedy', 'Pasutri gaje')) 

# Menyimpan perubahan
conn.commit()

# Menutup koneksi
conn.close()
