from mrjob.job import MRJob
from mrjob.step import MRStep
import re

class InvertedIndex(MRJob):
    def mapper(self, _, line):
        
        parts = line.strip().split('\t', 1)
        if len(parts) != 2:
            return
            
        doc_id, content = parts
        
        
        content = re.sub(r'[^a-z]+', ' ', content.lower())
        terms = content.split()
        
        # Emit (term, doc_id) pairs
        for term in terms:
            yield term, doc_id

    def reducer(self, term, doc_ids):
        
        term_counts = {}
        doc_ids_list = list(doc_ids)
        for doc_id in doc_ids_list:
            term_counts[doc_id] = term_counts.get(doc_id, 0) + 1
        
        
        frequency_list = " ".join(sorted([f"{doc}:{count}" for doc, count in term_counts.items()]))
        yield term, frequency_list

    def steps(self):
        return [MRStep(mapper=self.mapper, reducer=self.reducer)]

if __name__ == '__main__':
    InvertedIndex.run()
