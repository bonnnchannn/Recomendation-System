import sqlite3

# Membuat koneksi ke database SQLite
conn = sqlite3.connect('netflix.db')  # Ganti dengan nama file database Anda
cursor = conn.cursor()

# Data film dan genre baru

# Data film yang ingin dihapus
film_to_delete = [
    ('Scandal Makers',),
]

# Query untuk menghapus film berdasarkan judul
cursor.executemany("""
DELETE FROM movies
WHERE title = ?
""", film_to_delete)

# Menyimpan perubahan
conn.commit()

# Menutup koneksi
conn.close()
