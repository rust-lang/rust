use std::ops::Range;
use std::sync::Arc;

use crate::source_map::SourceMap;
use crate::{BytePos, Pos, RelativeBytePos, SourceFile, SpanData};

/// A `SourceMap` wrapper that caches info about a single recent code position. This gives a good
/// speedup when hashing spans, because often multiple spans within a single line are hashed in
/// succession, and this avoids expensive `SourceMap` lookups each time the cache is hit. We used
/// to cache multiple code positions, but caching a single position ended up being simpler and
/// faster.
pub struct CachingSourceMapView<'sm> {
    source_map: &'sm SourceMap,
    file: Arc<SourceFile>,
    // The line's byte position range in the `SourceMap`. This range will fail to contain a valid
    // position in certain edge cases. Spans often start/end one past something, and when that
    // something is the last character of a file (this can happen when a file doesn't end in a
    // newline, for example), we'd still like for the position to be considered within the last
    // line. However, it isn't according to the exclusive upper bound of this range. We cannot
    // change the upper bound to be inclusive, because for most lines, the upper bound is the same
    // as the lower bound of the next line, so there would be an ambiguity.
    //
    // Since the containment aspect of this range is only used to see whether or not the cache
    // entry contains a position, the only ramification of the above is that we will get cache
    // misses for these rare positions. A line lookup for the position via `SourceMap::lookup_line`
    // after a cache miss will produce the last line number, as desired.
    line_bounds: Range<BytePos>,
    line_number: usize,
}

impl<'sm> CachingSourceMapView<'sm> {
    pub fn new(source_map: &'sm SourceMap) -> CachingSourceMapView<'sm> {
        let files = source_map.files();
        let first_file = Arc::clone(&files[0]);
        CachingSourceMapView {
            source_map,
            file: first_file,
            line_bounds: BytePos(0)..BytePos(0),
            line_number: 0,
        }
    }

    #[inline]
    fn pos_to_line(&self, pos: BytePos) -> (Range<BytePos>, usize) {
        let pos = self.file.relative_position(pos);
        let line_index = self.file.lookup_line(pos).unwrap();
        let line_bounds = self.file.line_bounds(line_index);
        let line_number = line_index + 1;
        (line_bounds, line_number)
    }

    #[inline]
    fn update(&mut self, new_file: Option<Arc<SourceFile>>, pos: BytePos) {
        if let Some(file) = new_file {
            self.file = file;
        }
        (self.line_bounds, self.line_number) = self.pos_to_line(pos);
    }

    pub fn byte_pos_to_line_and_col(
        &mut self,
        pos: BytePos,
    ) -> Option<(Arc<SourceFile>, usize, RelativeBytePos)> {
        if self.line_bounds.contains(&pos) {
            // Cache hit: do nothing.
        } else {
            // Cache miss. If the entry doesn't point to the correct file, get the new file and
            // index.
            let new_file = if !file_contains(&self.file, pos) {
                Some(self.file_for_position(pos)?)
            } else {
                None
            };
            self.update(new_file, pos);
        };

        let col = RelativeBytePos(pos.to_u32() - self.line_bounds.start.to_u32());
        Some((Arc::clone(&self.file), self.line_number, col))
    }

    pub fn span_data_to_lines_and_cols(
        &mut self,
        span_data: &SpanData,
    ) -> Option<(&SourceFile, usize, BytePos, usize, BytePos)> {
        let lo_hit = self.line_bounds.contains(&span_data.lo);
        let hi_hit = self.line_bounds.contains(&span_data.hi);
        if lo_hit && hi_hit {
            // span_data.lo and span_data.hi are cached (i.e. both in the same line).
            return Some((
                &self.file,
                self.line_number,
                span_data.lo - self.line_bounds.start,
                self.line_number,
                span_data.hi - self.line_bounds.start,
            ));
        }

        // If the cached file is wrong, update it. Return early if the span lo and hi are in
        // different files.
        let new_file = if !file_contains(&self.file, span_data.lo) {
            let new_file = self.file_for_position(span_data.lo)?;
            if !file_contains(&new_file, span_data.hi) {
                return None;
            }
            Some(new_file)
        } else {
            if !file_contains(&self.file, span_data.hi) {
                return None;
            }
            None
        };

        // If we reach here, lo and hi are in the same file.

        if !lo_hit {
            // We cache the lo information.
            self.update(new_file, span_data.lo);
        }
        let lo_line_bounds = &self.line_bounds;
        let lo_line_number = self.line_number.clone();

        let (hi_line_bounds, hi_line_number) = if !self.line_bounds.contains(&span_data.hi) {
            // hi and lo are in different lines. We compute but don't cache the hi information.
            self.pos_to_line(span_data.hi)
        } else {
            // hi and lo are in the same line.
            (self.line_bounds.clone(), self.line_number)
        };

        // Span lo and hi may equal line end when last line doesn't
        // end in newline, hence the inclusive upper bounds below.
        assert!(span_data.lo >= lo_line_bounds.start);
        assert!(span_data.lo <= lo_line_bounds.end);
        assert!(span_data.hi >= hi_line_bounds.start);
        assert!(span_data.hi <= hi_line_bounds.end);
        assert!(self.file.contains(span_data.lo));
        assert!(self.file.contains(span_data.hi));

        Some((
            &self.file,
            lo_line_number,
            span_data.lo - lo_line_bounds.start,
            hi_line_number,
            span_data.hi - hi_line_bounds.start,
        ))
    }

    fn file_for_position(&self, pos: BytePos) -> Option<Arc<SourceFile>> {
        if !self.source_map.files().is_empty() {
            let file_idx = self.source_map.lookup_source_file_idx(pos);
            let file = &self.source_map.files()[file_idx];

            if file_contains(file, pos) {
                return Some(Arc::clone(file));
            }
        }

        None
    }
}

#[inline]
fn file_contains(file: &SourceFile, pos: BytePos) -> bool {
    // `SourceMap::lookup_source_file_idx` and `SourceFile::contains` both consider the position
    // one past the end of a file to belong to it. Normally, that's what we want. But for the
    // purposes of converting a byte position to a line and column number, we can't come up with a
    // line and column number if the file is empty, because an empty file doesn't contain any
    // lines. So for our purposes, we don't consider empty files to contain any byte position.
    file.contains(pos) && !file.is_empty()
}
