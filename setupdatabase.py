import sqlite3

# Membuat koneksi ke database SQLite
conn = sqlite3.connect('netflix.db')  # Ganti dengan nama file database Anda
cursor = conn.cursor()

# Data film yang ingin dimasukkan
film_data = [
    ('Layangan Putus: The Movie', 'Romance, Drama', 2023, 'Setelah resmi berpisah dari Aris, Kinan memulai perjalanan penemuan jati diri, menghadapi tantangan menjadi single parent sambil melanjutkan perannya sebagai dokter yang berdedikasi.', 'https://media.themoviedb.org/t/p/w600_and_h900_bestv2/kxhqebpoVDye8XDrWsLUpItucv5.jpg'),
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
