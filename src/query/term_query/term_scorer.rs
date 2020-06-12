use crate::docset::DocSet;
use crate::query::{Explanation, Scorer};
use crate::DocId;
use crate::Score;

use crate::fieldnorm::FieldNormReader;
use crate::postings::SegmentPostings;
use crate::postings::{FreqReadingOption, Postings};
use crate::query::bm25::BM25Weight;
use core::iter;

pub struct TermScorer {
    pub(crate) postings: SegmentPostings,
    fieldnorm_reader: FieldNormReader,
    similarity_weight: BM25Weight,
}

impl TermScorer {
    pub fn new(
        postings: SegmentPostings,
        fieldnorm_reader: FieldNormReader,
        similarity_weight: BM25Weight,
    ) -> TermScorer {
        TermScorer {
            postings,
            fieldnorm_reader,
            similarity_weight,
        }
    }

    #[cfg(test)]
    pub fn create_for_test(
        doc_and_tfs: &[(DocId, u32)],
        fieldnorm_vals: &[u32],
        similarity_weight: BM25Weight,
    ) -> TermScorer {
        assert!(!doc_and_tfs.is_empty());
        assert_eq!(doc_and_tfs.len(), fieldnorm_vals.len());
        let segment_postings = SegmentPostings::create_from_docs_and_tfs(doc_and_tfs);
        let doc_freq = doc_and_tfs.len();
        let max_doc = doc_and_tfs.last().unwrap().0 + 1;
        let mut fieldnorms: Vec<u32> = iter::repeat(0).take(max_doc as usize).collect();
        for i in 0..doc_freq {
            let doc = doc_and_tfs[i].0;
            let fieldnorm = fieldnorm_vals[i];
            fieldnorms[doc as usize] = fieldnorm;
        }
        let fieldnorm_reader = FieldNormReader::from(&fieldnorms[..]);
        TermScorer::new(segment_postings, fieldnorm_reader, similarity_weight)
    }

    pub(crate) fn freq_reading_option(&self) -> FreqReadingOption {
        self.postings.block_cursor.freq_reading_option()
    }

    pub fn block_max_score(&mut self) -> Score {
        self.postings
            .block_cursor
            .block_max_score(&self.fieldnorm_reader, &self.similarity_weight)
    }

    pub fn term_freq(&self) -> u32 {
        self.postings.term_freq()
    }

    pub fn doc_freq(&self) -> usize {
        self.postings.doc_freq() as usize
    }

    pub fn fieldnorm_id(&self) -> u8 {
        self.fieldnorm_reader.fieldnorm_id(self.doc())
    }

    pub fn explain(&self) -> Explanation {
        let fieldnorm_id = self.fieldnorm_id();
        let term_freq = self.term_freq();
        self.similarity_weight.explain(fieldnorm_id, term_freq)
    }

    pub fn max_score(&self) -> f32 {
        self.similarity_weight.max_score()
    }
}

impl DocSet for TermScorer {
    fn advance(&mut self) -> DocId {
        self.postings.advance()
    }

    fn seek(&mut self, target: DocId) -> DocId {
        self.postings.seek(target)
    }

    fn doc(&self) -> DocId {
        self.postings.doc()
    }

    fn size_hint(&self) -> u32 {
        self.postings.size_hint()
    }
}

impl Scorer for TermScorer {
    fn score(&mut self) -> Score {
        let fieldnorm_id = self.fieldnorm_id();
        let term_freq = self.term_freq();
        self.similarity_weight.score(fieldnorm_id, term_freq)
    }
}

#[cfg(test)]
mod tests {
    use crate::query::term_query::TermScorer;
    use crate::query::{BM25Weight, Scorer};
    use crate::tests::assert_nearly_equals;
    use crate::{DocSet, TERMINATED};

    #[test]
    fn test_term_scorer_max_score() {
        let bm25_weight = BM25Weight::for_one_term(3, 6, 10f32);
        let mut term_scorer =
            TermScorer::create_for_test(&[(2, 3), (3, 12), (7, 8)], &[10, 12, 100], bm25_weight);
        let max_scorer = term_scorer.max_score();
        assert_eq!(max_scorer, 1.3990127f32);
        assert_eq!(term_scorer.doc(), 2);
        assert_eq!(term_scorer.term_freq(), 3);
        assert_nearly_equals(term_scorer.block_max_score(), 1.3676447f32);
        assert_nearly_equals(term_scorer.score(), 1.0892314f32);
        assert_eq!(term_scorer.advance(), 3);
        assert_eq!(term_scorer.doc(), 3);
        assert_eq!(term_scorer.term_freq(), 12);
        assert_nearly_equals(term_scorer.score(), 1.3676447f32);
        assert_eq!(term_scorer.advance(), 7);
        assert_eq!(term_scorer.doc(), 7);
        assert_eq!(term_scorer.term_freq(), 8);
        assert_nearly_equals(term_scorer.score(), 0.72015285f32);
        assert_eq!(term_scorer.advance(), TERMINATED);
    }
}
