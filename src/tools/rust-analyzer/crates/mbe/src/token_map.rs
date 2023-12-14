//! Mapping between `TokenId`s and the token's position in macro definitions or inputs.

use std::hash::Hash;

use stdx::{always, itertools::Itertools};
use syntax::{TextRange, TextSize};
use tt::Span;

/// Maps absolute text ranges for the corresponding file to the relevant span data.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct SpanMap<S: Span> {
    spans: Vec<(TextSize, S)>,
}

impl<S: Span> SpanMap<S> {
    /// Creates a new empty [`SpanMap`].
    pub fn empty() -> Self {
        Self { spans: Vec::new() }
    }

    /// Finalizes the [`SpanMap`], shrinking its backing storage and validating that the offsets are
    /// in order.
    pub fn finish(&mut self) {
        always!(
            self.spans.iter().tuple_windows().all(|(a, b)| a.0 < b.0),
            "spans are not in order"
        );
        self.spans.shrink_to_fit();
    }

    /// Pushes a new span onto the [`SpanMap`].
    pub fn push(&mut self, offset: TextSize, span: S) {
        if cfg!(debug_assertions) {
            if let Some(&(last_offset, _)) = self.spans.last() {
                assert!(
                    last_offset < offset,
                    "last_offset({last_offset:?}) must be smaller than offset({offset:?})"
                );
            }
        }
        self.spans.push((offset, span));
    }

    /// Returns all [`TextRange`]s that correspond to the given span.
    ///
    /// Note this does a linear search through the entire backing vector.
    pub fn ranges_with_span(&self, span: S) -> impl Iterator<Item = TextRange> + '_ {
        // FIXME: This should ignore the syntax context!
        self.spans.iter().enumerate().filter_map(move |(idx, &(end, s))| {
            if s != span {
                return None;
            }
            let start = idx.checked_sub(1).map_or(TextSize::new(0), |prev| self.spans[prev].0);
            Some(TextRange::new(start, end))
        })
    }

    /// Returns the span at the given position.
    pub fn span_at(&self, offset: TextSize) -> S {
        let entry = self.spans.partition_point(|&(it, _)| it <= offset);
        self.spans[entry].1
    }

    /// Returns the spans associated with the given range.
    /// In other words, this will return all spans that correspond to all offsets within the given range.
    pub fn spans_for_range(&self, range: TextRange) -> impl Iterator<Item = S> + '_ {
        let (start, end) = (range.start(), range.end());
        let start_entry = self.spans.partition_point(|&(it, _)| it <= start);
        let end_entry = self.spans[start_entry..].partition_point(|&(it, _)| it <= end); // FIXME: this might be wrong?
        (&self.spans[start_entry..][..end_entry]).iter().map(|&(_, s)| s)
    }

    pub fn iter(&self) -> impl Iterator<Item = (TextSize, S)> + '_ {
        self.spans.iter().copied()
    }
}
