from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, explode, regexp_replace, lower, lit, row_number, concat, count, collect_list, concat_ws
from pyspark.sql.window import Window
import re
import sys

def build_bigram_index(input_file, output_file):
    # start spark
    spark = SparkSession.builder.appName("BigramIndex").getOrCreate()
    
    # read the input file
    lines = spark.read.text(input_file)
    
    def parse_document_line(line):
        # split on first tab to get doc id and text
        parts = line.split('\t', 1)
        if len(parts) == 2:
            return (parts[0], parts[1])
        else:
            return None
    
    # convert to rdd and parse lines
    rdd = lines.rdd.map(lambda row: row.value)
    docs = rdd.map(parse_document_line).filter(lambda x: x is not None)
    
    def get_bigrams_from_doc(doc_tuple):
        doc_id, text = doc_tuple
        
        # clean up the text - remove punctuation and make lowercase
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text)
        cleaned = cleaned.lower()
        
        # split into words and remove empty strings
        words = [w for w in cleaned.split() if w]
        
        # create bigrams
        bigrams = []
        for i in range(len(words) - 1):
            bigram = words[i] + ' ' + words[i + 1]
            bigrams.append((bigram, doc_id))
        
        return bigrams
    
    # extract all bigrams
    all_bigrams = docs.flatMap(get_bigrams_from_doc)
    
    # group by bigram and count frequencies per document
    def count_doc_frequencies(grouped_data):
        bigram, doc_ids = grouped_data
        
        # count how many times each document appears for this bigram
        doc_counts = {}
        for doc_id in doc_ids:
            if doc_id in doc_counts:
                doc_counts[doc_id] += 1
            else:
                doc_counts[doc_id] = 1
        
        # format as "docId:count" strings
        freq_strings = []
        for doc_id, count in doc_counts.items():
            freq_strings.append(f"{doc_id}:{count}")
        
        return (bigram, '\t'.join(freq_strings))
    
    # group and count
    grouped = all_bigrams.groupByKey()
    results = grouped.map(count_doc_frequencies)
    
    # convert back to dataframe for easier saving
    df = spark.createDataFrame(results, ["bigram", "frequencies"])
    
    # save to output
    df.coalesce(1).write.mode("overwrite").text(output_file)
    
    print("Done! Check the output directory.")
    spark.stop()

# alternative approach using sql operations
def build_bigram_index_with_sql(input_file, output_file):
    spark = SparkSession.builder.appName("BigramIndexSQL").getOrCreate()
    
    try:
        # read input
        df = spark.read.text(input_file)
        
        # parse the lines to get doc_id and text columns
        parsed = df.select(
            split(col("value"), "\t").getItem(0).alias("doc_id"),
            split(col("value"), "\t").getItem(1).alias("text")
        ).filter(col("text").isNotNull())
        
        # clean text and split into words
        words = parsed.select(
            col("doc_id"),
            explode(split(regexp_replace(lower(col("text")), "[^a-zA-Z]+", " "), " ")).alias("word")
        ).filter(col("word") != "")
        
        # add row numbers to create consecutive word pairs
        window = Window.partitionBy("doc_id").orderBy(lit(1))
        numbered_words = words.withColumn("word_index", row_number().over(window))
        
        # join with itself to create bigrams
        bigrams = numbered_words.alias("w1").join(
            numbered_words.alias("w2"),
            (col("w1.doc_id") == col("w2.doc_id")) & 
            (col("w1.word_index") == col("w2.word_index") - 1)
        ).select(
            col("w1.doc_id"),
            concat(col("w1.word"), lit(" "), col("w2.word")).alias("bigram")
        )
        
        # count frequencies and format output
        # this part is a bit tricky with the frequency counting
        freq_window = Window.partitionBy("bigram", "doc_id")
        bigram_counts = bigrams.withColumn("freq", count("*").over(freq_window))
        
        result = bigram_counts.groupBy("bigram").agg(
            collect_list(concat(col("doc_id"), lit(":"), col("freq"))).alias("doc_list")
        ).select(
            col("bigram"),
            concat_ws("\t", col("doc_list")).alias("frequencies")
        )
        
        # save it
        result.coalesce(1).write.mode("overwrite").text(output_file)
        
    except Exception as e:
        print(f"Something went wrong: {e}")
    finally:
        spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        exit(1)
    
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    
    # use the rdd approach by default
    build_bigram_index(input_path, output_path)
