from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
import re
import sys
from collections import defaultdict

def create_bigrams_inverted_index(input_path, output_path):
    spark = SparkSession.builder \
        .appName("BigramsInvertedIndex") \
        .getOrCreate()
    
    try:
        df = spark.read.text(input_path)
        
        def parse_line(line):
            parts = line.split('\t', 1)
            if len(parts) == 2:
                doc_id = parts[0]
                text = parts[1]
                return (doc_id, text)
            return None
        
        rdd = df.rdd.map(lambda row: row.value)
        parsed_rdd = rdd.map(parse_line).filter(lambda x: x is not None)
        
        def extract_bigrams(doc_data):
            doc_id, text = doc_data
            clean_text = re.sub(r'[^a-zA-Z]+', ' ', text).lower()
            words = [word for word in clean_text.split() if word.strip()]
            
            bigrams = []
            for i in range(len(words) - 1):
                bigram = f"{words[i]} {words[i+1]}"
                bigrams.append((bigram, doc_id))
            
            return bigrams
        
        bigrams_rdd = parsed_rdd.flatMap(extract_bigrams)
        
        def count_frequencies(bigram_docs):
            bigram, doc_ids = bigram_docs
            doc_freq = defaultdict(int)
            for doc_id in doc_ids:
                doc_freq[doc_id] += 1
            
            doc_freq_strings = [f"{doc_id}:{freq}" for doc_id, freq in doc_freq.items()]
            return (bigram, '\t'.join(doc_freq_strings))
        
        result_rdd = bigrams_rdd.groupByKey().map(count_frequencies)
        result_df = spark.createDataFrame(result_rdd, ["bigram", "doc_frequencies"])
        result_df.coalesce(1).write.mode("overwrite").text(output_path)
        
        print(f"Bigrams inverted index completed successfully!")
        print(f"Output written to: {output_path}")
        
    except Exception as e:
        print(f"Error processing bigrams inverted index: {str(e)}")
        raise
    finally:
        spark.stop()

def create_bigrams_inverted_index_sql(input_path, output_path):
    spark = SparkSession.builder \
        .appName("BigramsInvertedIndexSQL") \
        .getOrCreate()
    
    try:
        lines_df = spark.read.text(input_path)
        
        parsed_df = lines_df.select(
            split(col("value"), "\t").getItem(0).alias("doc_id"),
            split(col("value"), "\t").getItem(1).alias("text")
        ).filter(col("text").isNotNull())
        
        words_df = parsed_df.select(
            col("doc_id"),
            explode(
                split(
                    regexp_replace(lower(col("text")), "[^a-zA-Z]+", " "), 
                    " "
                )
            ).alias("word")
        ).filter(col("word") != "")
        
        from pyspark.sql.window import Window
        
        window_spec = Window.partitionBy("doc_id").orderBy(lit(1))
        words_with_index = words_df.withColumn("row_num", row_number().over(window_spec))
        
        bigrams_df = words_with_index.alias("w1").join(
            words_with_index.alias("w2"),
            (col("w1.doc_id") == col("w2.doc_id")) & 
            (col("w1.row_num") == col("w2.row_num") - 1)
        ).select(
            col("w1.doc_id"),
            concat(col("w1.word"), lit(" "), col("w2.word")).alias("bigram")
        )
        
        result_df = bigrams_df.groupBy("bigram") \
            .agg(
                collect_list(
                    concat(col("doc_id"), lit(":"), 
                           count("*").over(Window.partitionBy("bigram", "doc_id")))
                ).alias("doc_freq_list")
            ) \
            .select(
                col("bigram"),
                concat_ws("\t", col("doc_freq_list")).alias("doc_frequencies")
            )
        
        result_df.coalesce(1).write.mode("overwrite").text(output_path)
        
    finally:
        spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python bigrams_inverted_index.py <input_path> <output_path>", 
              file=sys.stderr)
        sys.exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    create_bigrams_inverted_index(input_path, output_path)
