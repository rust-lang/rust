//! A map that maps a span to every position in a file. Usually maps a span to some range of positions.
//! Allows bidirectional lookup.

use std::{fmt, hash::Hash};

use stdx::{always, itertools::Itertools};
use syntax::{TextRange, TextSize};
use vfs::FileId;

use crate::{
    ErasedFileAstId, Span, SpanAnchor, SpanData, SyntaxContextId, ROOT_ERASED_FILE_AST_ID,
};

/// Maps absolute text ranges for the corresponding file to the relevant span data.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct SpanMap<S> {
    spans: Vec<(TextSize, SpanData<S>)>,
}

impl<S> SpanMap<S>
where
    SpanData<S>: Copy,
{
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
    pub fn push(&mut self, offset: TextSize, span: SpanData<S>) {
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
    pub fn ranges_with_span_exact(&self, span: SpanData<S>) -> impl Iterator<Item = TextRange> + '_
    where
        S: Copy,
    {
        self.spans.iter().enumerate().filter_map(move |(idx, &(end, s))| {
            if !s.eq_ignoring_ctx(span) {
                return None;
            }
            let start = idx.checked_sub(1).map_or(TextSize::new(0), |prev| self.spans[prev].0);
            Some(TextRange::new(start, end))
        })
    }

    /// Returns all [`TextRange`]s whose spans contain the given span.
    ///
    /// Note this does a linear search through the entire backing vector.
    pub fn ranges_with_span(&self, span: SpanData<S>) -> impl Iterator<Item = TextRange> + '_
    where
        S: Copy,
    {
        self.spans.iter().enumerate().filter_map(move |(idx, &(end, s))| {
            if s.anchor != span.anchor {
                return None;
            }
            if !s.range.contains_range(span.range) {
                return None;
            }
            let start = idx.checked_sub(1).map_or(TextSize::new(0), |prev| self.spans[prev].0);
            Some(TextRange::new(start, end))
        })
    }

    /// Returns the span at the given position.
    pub fn span_at(&self, offset: TextSize) -> SpanData<S> {
        let entry = self.spans.partition_point(|&(it, _)| it <= offset);
        self.spans[entry].1
    }

    /// Returns the spans associated with the given range.
    /// In other words, this will return all spans that correspond to all offsets within the given range.
    pub fn spans_for_range(&self, range: TextRange) -> impl Iterator<Item = SpanData<S>> + '_ {
        let (start, end) = (range.start(), range.end());
        let start_entry = self.spans.partition_point(|&(it, _)| it <= start);
        let end_entry = self.spans[start_entry..].partition_point(|&(it, _)| it <= end); // FIXME: this might be wrong?
        self.spans[start_entry..][..end_entry].iter().map(|&(_, s)| s)
    }

    pub fn iter(&self) -> impl Iterator<Item = (TextSize, SpanData<S>)> + '_ {
        self.spans.iter().copied()
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct RealSpanMap {
    file_id: FileId,
    /// Invariant: Sorted vec over TextSize
    // FIXME: SortedVec<(TextSize, ErasedFileAstId)>?
    pairs: Box<[(TextSize, ErasedFileAstId)]>,
    end: TextSize,
}

impl fmt::Display for RealSpanMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RealSpanMap({:?}):", self.file_id)?;
        for span in self.pairs.iter() {
            writeln!(f, "{}: {}", u32::from(span.0), span.1.into_raw().into_u32())?;
        }
        Ok(())
    }
}

impl RealSpanMap {
    /// Creates a real file span map that returns absolute ranges (relative ranges to the root ast id).
    pub fn absolute(file_id: FileId) -> Self {
        RealSpanMap {
            file_id,
            pairs: Box::from([(TextSize::new(0), ROOT_ERASED_FILE_AST_ID)]),
            end: TextSize::new(!0),
        }
    }

    pub fn from_file(
        file_id: FileId,
        pairs: Box<[(TextSize, ErasedFileAstId)]>,
        end: TextSize,
    ) -> Self {
        Self { file_id, pairs, end }
    }

    pub fn span_for_range(&self, range: TextRange) -> Span {
        assert!(
            range.end() <= self.end,
            "range {range:?} goes beyond the end of the file {:?}",
            self.end
        );
        let start = range.start();
        let idx = self
            .pairs
            .binary_search_by(|&(it, _)| it.cmp(&start).then(std::cmp::Ordering::Less))
            .unwrap_err();
        let (offset, ast_id) = self.pairs[idx - 1];
        Span {
            range: range - offset,
            anchor: SpanAnchor { file_id: self.file_id, ast_id },
            ctx: SyntaxContextId::ROOT,
        }
    }
}
