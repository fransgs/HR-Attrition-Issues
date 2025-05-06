# Human Resources Issues in Jaya-Jaya Maju Company
## Business Understanding
Jaya Jaya Maju merupakan salah satu perusahaan multinasional yang telah berdiri sejak tahun 2000. Ia memiliki lebih dari 1000 karyawan yang tersebar di seluruh penjuru negeri. Walaupun telah menjadi menjadi perusahaan yang cukup besar, Jaya Jaya Maju masih cukup kesulitan dalam mengelola karyawan. Hal ini berimbas tingginya attrition rate (rasio jumlah karyawan yang keluar dengan total karyawan keseluruhan) hingga lebih dari 10%. Untuk mencegah hal ini semakin parah, manajer departemen HR ingin meminta bantuan Anda mengidentifikasi berbagai faktor yang mempengaruhi tingginya attrition rate tersebut. Selain itu, ia juga meminta Anda untuk membuat business dashboard untuk membantunya memonitori berbagai faktor tersebut.

### Business Problem
Masalah bisnis utama yang dihadapi oleh perusahaan Jaya Jaya Maju adalah tingginya attrition rate, yang saat ini mencapai sekitar 17%, jauh di atas ambang batas ideal 10%. Tingginya angka ini mengindikasikan bahwa jumlah karyawan yang memutuskan untuk mengundurkan diri jauh lebih tinggi dibandingkan dengan yang tetap bertahan. Kondisi ini memberikan dampak langsung terhadap stabilitas operasional perusahaan, terutama pada departemen Research & Development dan job role seperti Laboratory Technician dan Sales Executive, yang mencatat tingkat pengunduran diri tertinggi.

Attrition yang tinggi ini menyebabkan terjadinya ketimpangan beban kerja, di mana karyawan yang tersisa harus menanggung tugas tambahan dari posisi yang kosong. Hal ini berpotensi menurunkan semangat kerja, meningkatkan stres, dan pada akhirnya menurunkan produktivitas tim. Dampaknya bisa menjalar lebih jauh, memengaruhi pencapaian target perusahaan dan menciptakan ketidakstabilan dalam operasional jangka panjang.

### Project Pipeline
1. Data preparation and data cleansing untuk mengumpulkan, memformat, dan membersihkan data
2. Exploratory Data Analysis (EDA) untuk mengeksplorasi dan memahami struktur serta pola data menggunakan statistik dan visualisasi
3. Membuat dashboard bisnis untuk visualisasi data dalam bentuk dashboard interaktif yang mudah dipahami oleh pengguna non-teknis
4. Membuat dan membangun model machine learning untuk prediksi attrition di masa depan

Proyek ini menggunakan dataset sebagai berikut: [Employee HR Dataset](https://github.com/dicodingacademy/dicoding_dataset/tree/main/employee)

## Business Dashboard
Dashboard dibuat secara sederhana namun tidak menghilangkan esensi dari informasi informasi yang diberikan kepada stakeholder HR terkait attrition rate yang mencapai 17% dan faktor-faktor apa saja yang menyebabkan attrition tersebut terjadi.

## Conclusion
Berdasarkan visualisasi data ini, dapat disimpulkan bahwa:
- Tingkat attrition tertinggi terjadi di usia produktif, terutama laki-laki usia 26–35 tahun.
- Peran pekerjaan yang bersifat teknis dan departemen sales mengalami tingkat keluar tertinggi.
- Mayoritas yang resign justru berasal dari kelompok berkinerja tinggi.
- Faktor lain yang memengaruhi adalah pengalaman kerja yang minim, jarak tempat tinggal, dan kurangnya keterlibatan dalam kegiatan bisnis atau pengembangan diri.

### Recommendation Action Items
Berikut ini rekomendasi yang dapat dilakukan untuk manajemen adalah sebagai berikut:
- Fokus pada peningkatan retensi di usia muda, terutama dalam departemen Sales dan peran teknis.
- Evaluasi ulang beban kerja, program pengembangan karier, dan pemberian insentif untuk karyawan berkinerja tinggi.
- Tingkatkan peluang pelatihan dan business engagement untuk semua level karyawan.
- Lakukan survei internal berkala untuk menggali lebih dalam alasan resign yang tidak terlihat di data.
- Prediksi kecenderungan karyawan melakukan resign dapat dilakukan melalui website berikut: [Attrition Prediction](https://attrition-prediction-company.streamlit.app/)
