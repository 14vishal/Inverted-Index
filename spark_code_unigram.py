from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, explode, regexp_replace, lower, lit, count, collect_list, concat_ws
import sys

def build_unigram_index(input_file, output_file):
    spark = SparkSession.builder.appName("UnigramIndex").getOrCreate()

    df = spark.read.text(input_file)

    parsed = df.select(
        split(col("value"), "\t").getItem(0).alias("doc_id"),
        split(col("value"), "\t").getItem(1).alias("text")
    ).filter(col("text").isNotNull())

    words = parsed.select(
        col("doc_id"),
        explode(split(regexp_replace(lower(col("text")), "[^a-zA-Z]+", " "), " ")).alias("word")
    ).filter(col("word") != "")

    unigram_counts = words.groupBy("word", "doc_id").agg(count("word").alias("freq"))

    result = unigram_counts.groupBy("word").agg(
        collect_list(concat_ws(":", col("doc_id"), col("freq"))).alias("doc_list")
    ).select(
        col("word"),
        concat_ws("\t", col("doc_list")).alias("frequencies")
    )

    result.coalesce(1).write.mode("overwrite").text(output_file)

    print("Done! Check the output directory.")
    spark.stop()

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_path> <output_path>")
        exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    build_unigram_index(input_path, output_path)
