//! Mapping between `TokenId`s and the token's position in macro definitions or inputs.

use std::hash::Hash;

use stdx::itertools::Itertools;
use syntax::{TextRange, TextSize};
use tt::Span;

/// Maps absolute text ranges for the corresponding file to the relevant span data.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
// FIXME: Rename to SpanMap
pub struct TokenMap<S: Span> {
    // FIXME: This needs to be sorted by (FileId, AstId)
    // Then we can do a binary search on the file id,
    // then a bin search on the ast id?
    spans: Vec<(TextSize, S)>,
}

impl<S: Span> TokenMap<S> {
    pub fn empty() -> Self {
        Self { spans: Vec::new() }
    }

    pub fn finish(&mut self) {
        assert!(self.spans.iter().tuple_windows().all(|(a, b)| a.0 < b.0));
        self.spans.shrink_to_fit();
    }

    pub fn push(&mut self, offset: TextSize, span: S) {
        self.spans.push((offset, span));
    }

    pub fn ranges_with_span(&self, span: S) -> impl Iterator<Item = TextRange> + '_ {
        // FIXME: linear search
        self.spans.iter().enumerate().filter_map(move |(idx, &(end, s))| {
            if s != span {
                return None;
            }
            let start = idx.checked_sub(1).map_or(TextSize::new(0), |prev| self.spans[prev].0);
            Some(TextRange::new(start, end))
        })
    }

    // FIXME: We need APIs for fetching the span of a token as well as for a whole node. The node
    // one *is* fallible though.
    pub fn span_at(&self, offset: TextSize) -> S {
        let entry = self.spans.partition_point(|&(it, _)| it <= offset);
        self.spans[entry].1
    }

    pub fn spans_for_node_range(&self, range: TextRange) -> impl Iterator<Item = S> + '_ {
        let (start, end) = (range.start(), range.end());
        let start_entry = self.spans.partition_point(|&(it, _)| it <= start);
        let end_entry = self.spans[start_entry..].partition_point(|&(it, _)| it <= end); // FIXME: this might be wrong?
        (&self.spans[start_entry..][..end_entry]).iter().map(|&(_, s)| s)
    }

    pub fn iter(&self) -> impl Iterator<Item = (TextSize, S)> + '_ {
        self.spans.iter().copied()
    }
}
