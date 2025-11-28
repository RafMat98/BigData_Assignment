
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg, concat, lit, round as spark_round 
import builtins 
import time
import statistics
import os

# Προσθήκη στην αρχή του flight_delay_spark.py Χρήση για το Θέμα 3
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------
# 0. ΡΥΘΜΙΣΕΙΣ & ΠΡΟΕΤΟΙΜΑΣΙΑ
# ------------------------------------------------------------------------------
DATA_PATH = "flights_2000.csv"
OUTPUT_PATH = "out/top_10_routes_df"
NUM_RUNS = 5

# Εκκίνηση Spark Session & Context
spark = SparkSession.builder \
    .appName("FlightDelayAnalysis_Combined") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

print("--------------------------------------------------")
print(f"Ανάλυση Δεδομένων Πτήσεων με Apache Spark ({spark.version})")
print(f"Εκτέλεση: {NUM_RUNS} επαναλήψεις ανά API")
print("--------------------------------------------------\n")


# ------------------------------------------------------------------------------
# A. ΘΕΜΑ 1: SPARK RDD API (Μέση Καθυστέρηση ανά Αεροδρόμιο)
# ------------------------------------------------------------------------------

# ORIGIN_AIRPORT (index 3), DEP_DELAY (index 6), CANCELLED (index 10)
def clean_and_extract_rdd(row):
    """Διαχωρίζει τη γραμμή, καθαρίζει και επιστρέφει (OriginAirport, DepDelay)"""
    try:
        fields = row.split(',') 
        
        if len(fields) < 11:
            return None
            
        origin_airport = fields[3]
        dep_delay_str = fields[6]
        cancelled_str = fields[10]
        
        # 1. Εξαίρεση ακυρωμένων πτήσεων (CANCELLED = 1)
        if cancelled_str == '1':
            return None
        
        # 2. Εξαίρεση πτήσεων με έλλειψη δεδομένων καθυστέρησης
        if dep_delay_str in ('NA', '', 'null'): 
            return None
            
        dep_delay = float(dep_delay_str)
        
        return (origin_airport, dep_delay)
    
    except (ValueError, IndexError):
        return None

def calculate_top_10_delays_rdd():
    """Εκτελεί όλη τη λογική του RDD."""
    lines_rdd = sc.textFile(DATA_PATH) 
    header = lines_rdd.first() 
    data_rdd = lines_rdd.filter(lambda row: row != header) 

    airport_delays_rdd = data_rdd \
        .map(clean_and_extract_rdd) \
        .filter(lambda x: x is not None) 

    # Υπολογισμός μέσης καθυστέρησης με reduceByKey (Άθροισμα, Πλήθος)
    sum_count_rdd = airport_delays_rdd.mapValues(lambda x: (x, 1))

    sum_count_by_airport = sum_count_rdd.reduceByKey(
        lambda a, b: (a[0] + b[0], a[1] + b[1])
    )

    avg_delay_by_airport = sum_count_by_airport.mapValues(
        lambda sum_count: sum_count[0] / sum_count[1]
    )
    
    # Εύρεση top-10
    sorted_rdd = avg_delay_by_airport \
        .map(lambda x: (x[1], x[0])) \
        .sortByKey(ascending=False)

    top_10 = sorted_rdd.take(10) # <-- ACTION
    
    return [(airport, builtins.round(delay, 4)) for delay, airport in top_10]


# --- Επαναληπτική Εκτέλεση & Μέτρηση Χρόνου RDD ---
total_times_rdd = []
action_times_rdd = []
sample_output_rdd = None

print("--- Εκτέλεση RDD API (Θέμα 1) ---")
for i in range(NUM_RUNS):
    total_start = time.time()
    
    # Χρόνος Action
    action_start = time.time()
    result = calculate_top_10_delays_rdd()
    action_end = time.time()
    
    total_end = time.time()
    
    total_times_rdd.append(total_end - total_start)
    action_times_rdd.append(action_end - action_start)
    sample_output_rdd = result
    
    print(f"Run {i+1}: Total Time = {total_times_rdd[-1]:.4f}s, Action Time = {action_times_rdd[-1]:.4f}s")

# Υπολογισμός Μέσου Όρου RDD
if NUM_RUNS >= 3:
    avg_total_time_rdd = statistics.mean(sorted(total_times_rdd)[1:-1])
    avg_action_time_rdd = statistics.mean(sorted(action_times_rdd)[1:-1])
else:
    avg_total_time_rdd = statistics.mean(total_times_rdd)
    avg_action_time_rdd = statistics.mean(action_times_rdd)

print("\n")

# ------------------------------------------------------------------------------
# B. ΘΕΜΑ 2: SPARK DATAFRAME API (Μέση Καθυστέρηση ανά Διαδρομή)
# ------------------------------------------------------------------------------

# 1. Διαβάστε το αρχείο Flights.csv ως DataFrame
flights_df = spark.read.csv(DATA_PATH, header=True, inferSchema=True)

# 2. Εμφανίστε το schema του πίνακα (μόνο μία φορά)
print("--- DataFrame Schema (για επαλήθευση) ---")
flights_df.printSchema()
print("\n")


def calculate_top_10_routes_df(df):
    """Εκτελεί τη λογική του DataFrame API."""
    
    # 3. Φιλτράρισμα & Καθαρισμός
    # Φιλτράρισμα: CANCELLED = 0
    cleaned_df = df.filter(col("CANCELLED") == 0)

    # Αγνοήστε κενές ή μη αριθμητικές τιμές DEP_DELAY
    cleaned_df = cleaned_df.na.drop(subset=["DEP_DELAY"])

    # Δημιουργία της στήλης 'ROUTE' (Origin -> Destination)
    df_with_route = cleaned_df.withColumn(
        "ROUTE",
        concat(col("ORIGIN_AIRPORT"), lit("->"), col("DEST_AIRPORT"))
    )

    # 3a. Υπολογισμός μέσης καθυστέρησης αναχώρησης ανά διαδρομή (Route)
    avg_delay_by_route = df_with_route.groupBy("ROUTE").agg(
    spark_round(avg("DEP_DELAY"), 4).alias("AVG_DEP_DELAY_MIN") # Αλλαγή εδώ
    )

    # 3b. Τα 10 routes με τη μεγαλύτερη μέση καθυστέρηση (φθίνουσα)
    top_10_routes_df = avg_delay_by_route.orderBy(
        col("AVG_DEP_DELAY_MIN").desc()
    ).limit(10)

    return top_10_routes_df



# ------------------------------------------------------------------------------
# Δ. ΘΕΜΑ 4: ΟΠΤΙΚΟΠΟΙΗΣΗ ΤΟP-10 ROUTES
# ------------------------------------------------------------------------------

def visualize_top_10_routes(spark_df, output_filename):
    """
    Μετατρέπει το Spark DataFrame σε Pandas DataFrame και δημιουργεί οριζόντιο ραβδόγραμμα.
    """
    
    # 1. Μετατροπή σε Pandas DataFrame
    # Χρησιμοποιούμε .toPandas() για να φέρουμε τα 10 αποτελέσματα στη μνήμη του driver.
    # Αυτό είναι ασφαλές καθώς πρόκειται για πολύ μικρό σύνολο δεδομένων (10 γραμμές).
    pandas_df = spark_df.toPandas()

    # Ταξινόμηση για να είναι το μεγαλύτερο στην κορυφή του γραφήματος
    pandas_df = pandas_df.sort_values(by="AVG_DEP_DELAY_MIN", ascending=True)

    # 2. & 3. Δημιουργία Οριζόντιου Ραβδογράμματος (Horizontal Bar Chart)
    
    plt.figure(figsize=(10, 6))
    plt.barh(
        pandas_df['ROUTE'], 
        pandas_df['AVG_DEP_DELAY_MIN'], 
        color='skyblue'
    )
    
    # Προσθήκη τίτλων και ετικετών
    plt.title('Top-10 Διαδρομές με τη Μεγαλύτερη Μέση Καθυστέρηση Αναχώρησης')
    plt.xlabel('Μέση Καθυστέρηση Αναχώρησης (λεπτά)')
    plt.ylabel('Διαδρομή (ORIGIN->DEST)')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Προσθήκη των τιμών δίπλα σε κάθε μπάρα
    for index, value in enumerate(pandas_df['AVG_DEP_DELAY_MIN']):
        plt.text(value, index, f'{value:.2f}', va='center')
        
    plt.tight_layout() # Προσαρμογή διάταξης
    
    # Αποθήκευση του γραφήματος στον φάκελο εξόδου
    full_output_path = os.path.join("out", output_filename)
    plt.savefig(full_output_path)
    plt.close()
    
    print(f"✅ Δημιουργήθηκε γράφημα και αποθηκεύτηκε στο: {full_output_path}")
    
    

# --- Επαναληπτική Εκτέλεση & Μέτρηση Χρόνου DataFrame ---
total_times_df = []
action_times_df = []
final_df_result = None

print("--- Εκτέλεση DataFrame API (Θέμα 2) ---")
for i in range(NUM_RUNS):
    total_start = time.time()
    
    top_10_df = calculate_top_10_routes_df(flights_df)

    # Χρόνος Action: collect() αναγκάζει την εκτέλεση
    action_start = time.time()
    result_list = top_10_df.collect()
    action_end = time.time()

    total_end = time.time()

    total_times_df.append(total_end - total_start)
    action_times_df.append(action_end - action_start)
    final_df_result = top_10_df # Κρατάμε το DataFrame για αποθήκευση/εμφάνιση
    
    print(f"Run {i+1}: Total Time = {total_times_df[-1]:.4f}s, Action Time = {action_times_df[-1]:.4f}s")

# Υπολογισμός Μέσου Όρου DataFrame
if NUM_RUNS >= 3:
    avg_total_time_df = statistics.mean(sorted(total_times_df)[1:-1])
    avg_action_time_df = statistics.mean(sorted(action_times_df)[1:-1])
else:
    avg_total_time_df = statistics.mean(total_times_df)
    avg_action_time_df = statistics.mean(action_times_df)


# 4. Αποθήκευση του τελικού DataFrame σε CSV (Θέμα 2)
if final_df_result:
    # Πρώτα διαγράφουμε τον φάκελο εξόδου αν υπάρχει (για καθαρό output)
    print(f"\nΑποθήκευση Top-10 Routes στο {OUTPUT_PATH}...")
    final_df_result.write.mode('overwrite').csv(OUTPUT_PATH, header=True)
    print("✅ Αποθήκευση Ολοκληρώθηκε!")

# ------------------------------------------------------------------------------
# Ε. ΘΕΜΑ 5: ΠΡΟΣΘΕΤΗ ΑΝΑΛΥΣΗ (Μέση Καθυστέρηση ανά Ώρα)
# ------------------------------------------------------------------------------

def calculate_avg_delay_by_hour(df):
    """
    Υπολογίζει τη μέση καθυστέρηση αναχώρησης ανά ώρα της ημέρας.
    """
    
    # Καθαρισμός: Όπως και πριν, φιλτράρουμε ακυρωμένες πτήσεις και κενές καθυστερήσεις
    cleaned_df = df.filter(col("CANCELLED") == 0).na.drop(subset=["DEP_DELAY", "SCHED_DEP"])

    # 1. Δημιουργία στήλης 'HOUR' από το SCHED_DEP
    # Το SCHED_DEP είναι timestamp, οπότε χρησιμοποιούμε hour() ή substr.
    # Επειδή το schema σας δείχνει SCHED_DEP ως timestamp, μπορούμε να χρησιμοποιήσουμε την
    # ενσωματωμένη συνάρτηση hour() για μεγαλύτερη ασφάλεια.
    
    from pyspark.sql.functions import hour
    
    df_with_hour = cleaned_df.withColumn(
        "HOUR", 
        hour(col("SCHED_DEP"))
    )

    # 2. Υπολογισμός μέσης καθυστέρησης ανά ώρα
    avg_delay_by_hour = df_with_hour.groupBy("HOUR").agg(
        spark_round(avg("DEP_DELAY"), 2).alias("AVG_DELAY")
    )

    # 3. Ταξινόμηση ανά ώρα (0-23) για το γράφημα
    sorted_result = avg_delay_by_hour.orderBy(col("HOUR").asc())

    return sorted_result



# ------------------------------------------------------------------------------
# ΣΤ. ΟΠΤΙΚΟΠΟΙΗΣΗ ΘΕΜΑΤΟΣ 5
# ------------------------------------------------------------------------------

def visualize_delay_by_hour(spark_df, output_filename):
    """
    Δημιουργεί γραμμικό διάγραμμα μέσης καθυστέρησης ανά ώρα αναχώρησης.
    """
    
    # Μετατροπή σε Pandas DataFrame
    pandas_df = spark_df.toPandas()

    plt.figure(figsize=(12, 6))
    
    # Γραμμικό Διάγραμμα
    plt.plot(
        pandas_df['HOUR'], 
        pandas_df['AVG_DELAY'], 
        marker='o', 
        linestyle='-', 
        color='darkred'
    )
    
    # Προσθήκη τίτλων και ετικετών
    plt.title('Μέση Καθυστέρηση Αναχώρησης ανά Ώρα της Ημέρας', fontsize=14)
    plt.xlabel('Ώρα Αναχώρησης (24ωρη βάση)', fontsize=12)
    plt.ylabel('Μέση Καθυστέρηση (λεπτά)', fontsize=12)
    plt.xticks(range(0, 24)) # Εμφάνιση όλων των ωρών
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout() 
    
    # Αποθήκευση του γραφήματος
    full_output_path = os.path.join("out", output_filename)
    plt.savefig(full_output_path)
    plt.close()
    
    print(f"✅ Δημιουργήθηκε γράφημα πρόσθετης ανάλυσης στο: {full_output_path}")
    

# ------------------------------------------------------------------------------
# Γ. ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ & ΣΥΝΟΨΗ
# ------------------------------------------------------------------------------

print("\n" + "="*70)
print("                       ΤΕΛΙΚΑ ΑΠΟΤΕΛΕΣΜΑΤΑ & ΧΡΟΝΟΙ")
print("="*70)

# 1. Αποτελέσματα RDD (Θέμα 1)
print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ RDD API (Top-10 Αεροδρόμια) ---")
print(f"{'Αεροδρόμιο':<15} {'Μέση Καθυστέρηση (λεπτά)':>30}")
print("-" * 45)
for airport, delay in sample_output_rdd:
    print(f"{airport:<15} {delay:>30.4f}")

print(f"\n**Μέσος Όρος (3 runs) RDD Total Time:** {avg_total_time_rdd:.4f}s")
print(f"**Μέσος Όρος (3 runs) RDD Action Time:** {avg_action_time_rdd:.4f}s")


# 2. Αποτελέσματα DataFrame (Θέμα 2)
print("\n--- ΑΠΟΤΕΛΕΣΜΑΤΑ DATAFRAME API (Top-10 Διαδρομές) ---")
final_df_result.show(truncate=False)

print(f"\n**Μέσος Όρος (3 runs) DF Total Time:** {avg_total_time_df:.4f}s")
print(f"**Μέσος Όρος (3 runs) DF Action Time:** {avg_action_time_df:.4f}s")


# ------------------------------------------------------------------------------
# ΚΛΗΣΗ ΓΙΑ ΟΠΤΙΚΟΠΟΙΗΣΗ (ΘΕΜΑ 4)
# ------------------------------------------------------------------------------
if final_df_result is not None:
    visualize_top_10_routes(final_df_result, "top_10_delays_bar.png")


# ΚΛΗΣΗ ΓΙΑ ΠΡΟΣΘΕΤΗ ΑΝΑΛΥΣΗ & ΟΠΤΙΚΟΠΟΙΗΣΗ (ΘΕΜΑ 5)
# ------------------------------------------------------------------------------
print("\n--- Εκτέλεση Πρόσθετης Ανάλυσης (Θέμα 5) ---")
avg_delay_by_hour_df = calculate_avg_delay_by_hour(flights_df)

# Εμφάνιση ενός δείγματος του πίνακα (προαιρετικό)
print("Μέση Καθυστέρηση ανά Ώρα:")
avg_delay_by_hour_df.show(5)

# Οπτικοποίηση
visualize_delay_by_hour(avg_delay_by_hour_df, "avg_delay_by_hour_line.png")
    
# Τερματισμός Spark
spark.stop()

# ΣΗΜΕΙΩΣΗ: ΠΡΟΣΘΗΚΗ ΤΟΥ ΘΕΜΑΤΟΣ 3 (Γραφήματα & Σύγκριση) ΣΤΗΝ ΑΝΑΦΟΡΑ PDF!