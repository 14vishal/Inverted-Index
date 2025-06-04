from mrjob.job import MRJob
import re

TARGET_BIGRAMS = {
    'computer science',
    'information retrieval',
    'power politics',
    'los angeles',
    'bruce willis'
}

class MRBigramInvertedIndex(MRJob):

    def mapper(self, _, line):
        
        if '\t' in line:
            doc_id, content = line.split('\t', 1)
            
            
            content_clean = re.sub('[^a-z]+', ' ', content.lower())
            terms = content_clean.split()
            
            
            for i in range(len(terms) - 1):
                bigram = f"{terms[i]} {terms[i+1]}"
                if bigram in TARGET_BIGRAMS:
                    yield bigram, doc_id

    def reducer(self, bigram, doc_ids):
        
        doc_counts = {}
        for doc_id in doc_ids:
            doc_counts[doc_id] = doc_counts.get(doc_id, 0) + 1
        
        
        result = ' '.join([f"{k}:{v}" for k, v in doc_counts.items()])
        yield bigram, result

if __name__ == '__main__':
    MRBigramInvertedIndex.run()
