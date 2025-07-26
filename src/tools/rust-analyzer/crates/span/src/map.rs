//! A map that maps a span to every position in a file. Usually maps a span to some range of positions.
//! Allows bidirectional lookup.

use std::{fmt, hash::Hash};

use stdx::{always, itertools::Itertools};

use crate::{
    EditionedFileId, ErasedFileAstId, ROOT_ERASED_FILE_AST_ID, Span, SpanAnchor, SpanData,
    SyntaxContext, TextRange, TextSize,
};

/// Maps absolute text ranges for the corresponding file to the relevant span data.
#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub struct SpanMap<S> {
    /// The offset stored here is the *end* of the node.
    spans: Vec<(TextSize, SpanData<S>)>,
    /// Index of the matched macro arm on successful expansion for declarative macros.
    // FIXME: Does it make sense to have this here?
    pub matched_arm: Option<u32>,
}

impl<S> SpanMap<S>
where
    SpanData<S>: Copy,
{
    /// Creates a new empty [`SpanMap`].
    pub fn empty() -> Self {
        Self { spans: Vec::new(), matched_arm: None }
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
    pub fn ranges_with_span_exact(
        &self,
        span: SpanData<S>,
    ) -> impl Iterator<Item = (TextRange, S)> + '_
    where
        S: Copy,
    {
        self.spans.iter().enumerate().filter_map(move |(idx, &(end, s))| {
            if !s.eq_ignoring_ctx(span) {
                return None;
            }
            let start = idx.checked_sub(1).map_or(TextSize::new(0), |prev| self.spans[prev].0);
            Some((TextRange::new(start, end), s.ctx))
        })
    }

    /// Returns all [`TextRange`]s whose spans contain the given span.
    ///
    /// Note this does a linear search through the entire backing vector.
    pub fn ranges_with_span(&self, span: SpanData<S>) -> impl Iterator<Item = (TextRange, S)> + '_
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
            Some((TextRange::new(start, end), s.ctx))
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

    /// Merges this span map with another span map, where `other` is inserted at (and replaces) `other_range`.
    ///
    /// The length of the replacement node needs to be `other_size`.
    pub fn merge(&mut self, other_range: TextRange, other_size: TextSize, other: &SpanMap<S>) {
        // I find the following diagram helpful to illustrate the bounds and why we use `<` or `<=`:
        // --------------------------------------------------------------------
        //   1   3   5   6   7   10    11          <-- offsets we store
        // 0-1 1-3 3-5 5-6 6-7 7-10 10-11          <-- ranges these offsets refer to
        //       3   ..      7                     <-- other_range
        //         3-5 5-6 6-7                     <-- ranges we replace (len = 7-3 = 4)
        //         ^^^^^^^^^^^ ^^^^^^^^^^
        //           remove       shift
        //   2   3   5   9                         <-- offsets we insert
        // 0-2 2-3 3-5 5-9                         <-- ranges we insert (other_size = 9-0 = 9)
        // ------------------------------------
        //   1   3
        // 0-1 1-3                                 <-- these remain intact
        //           5   6   8   12
        //         3-5 5-6 6-8 8-12                <-- we shift these by other_range.start() and insert them
        //                             15    16
        //                          12-15 15-16    <-- we shift these by other_size-other_range.len() = 9-4 = 5
        // ------------------------------------
        //   1   3   5   6   8   12    15    16    <-- final offsets we store
        // 0-1 1-3 3-5 5-6 6-8 8-12 12-15 15-16    <-- final ranges

        self.spans.retain_mut(|(offset, _)| {
            if other_range.start() < *offset && *offset <= other_range.end() {
                false
            } else {
                if *offset > other_range.end() {
                    *offset += other_size;
                    *offset -= other_range.len();
                }
                true
            }
        });

        self.spans
            .extend(other.spans.iter().map(|&(offset, span)| (offset + other_range.start(), span)));

        self.spans.sort_unstable_by_key(|&(offset, _)| offset);

        // Matched arm info is no longer correct once we have multiple macros.
        self.matched_arm = None;
    }
}

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct RealSpanMap {
    file_id: EditionedFileId,
    /// Invariant: Sorted vec over TextSize
    // FIXME: SortedVec<(TextSize, ErasedFileAstId)>?
    pairs: Box<[(TextSize, ErasedFileAstId)]>,
    end: TextSize,
}

impl fmt::Display for RealSpanMap {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "RealSpanMap({:?}):", self.file_id)?;
        for span in self.pairs.iter() {
            writeln!(f, "{}: {:#?}", u32::from(span.0), span.1)?;
        }
        Ok(())
    }
}

impl RealSpanMap {
    /// Creates a real file span map that returns absolute ranges (relative ranges to the root ast id).
    pub fn absolute(file_id: EditionedFileId) -> Self {
        RealSpanMap {
            file_id,
            pairs: Box::from([(TextSize::new(0), ROOT_ERASED_FILE_AST_ID)]),
            end: TextSize::new(!0),
        }
    }

    pub fn from_file(
        file_id: EditionedFileId,
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
            ctx: SyntaxContext::root(self.file_id.edition()),
        }
    }
}
