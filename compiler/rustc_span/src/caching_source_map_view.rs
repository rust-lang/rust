use std::ops::Range;

use rustc_data_structures::sync::Lrc;

use crate::source_map::SourceMap;
use crate::{BytePos, Pos, RelativeBytePos, SourceFile, SpanData, StableSourceFileId};

#[derive(Clone)]
struct CacheEntry {
    time_stamp: usize,
    line_number: usize,
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
    line: Range<BytePos>,
    file: Lrc<SourceFile>,
    file_index: usize,
}

impl CacheEntry {
    #[inline]
    fn update(
        &mut self,
        new_file_and_idx: Option<(Lrc<SourceFile>, usize)>,
        pos: BytePos,
        time_stamp: usize,
    ) {
        if let Some((file, file_idx)) = new_file_and_idx {
            self.file = file;
            self.file_index = file_idx;
        }

        let pos = self.file.relative_position(pos);
        let line_index = self.file.lookup_line(pos).unwrap();
        let line_bounds = self.file.line_bounds(line_index);
        self.line_number = line_index + 1;
        self.line = line_bounds;
        self.touch(time_stamp);
    }

    #[inline]
    fn touch(&mut self, time_stamp: usize) {
        self.time_stamp = time_stamp;
    }
}

#[derive(Clone)]
pub struct CachingSourceMapView<'sm> {
    source_map: &'sm SourceMap,
    line_cache: [CacheEntry; 3],
    time_stamp: usize,
}

impl<'sm> CachingSourceMapView<'sm> {
    pub fn new(source_map: &'sm SourceMap) -> CachingSourceMapView<'sm> {
        let files = source_map.files();
        let first_file = Lrc::clone(&files[0]);
        let entry = CacheEntry {
            time_stamp: 0,
            line_number: 0,
            line: BytePos(0)..BytePos(0),
            file: first_file,
            file_index: 0,
        };

        CachingSourceMapView {
            source_map,
            line_cache: [entry.clone(), entry.clone(), entry],
            time_stamp: 0,
        }
    }

    pub fn byte_pos_to_line_and_col(
        &mut self,
        pos: BytePos,
    ) -> Option<(Lrc<SourceFile>, usize, RelativeBytePos)> {
        self.time_stamp += 1;

        // Check if the position is in one of the cached lines
        let cache_idx = self.cache_entry_index(pos);
        if cache_idx != -1 {
            let cache_entry = &mut self.line_cache[cache_idx as usize];
            cache_entry.touch(self.time_stamp);

            let col = RelativeBytePos(pos.to_u32() - cache_entry.line.start.to_u32());
            return Some((Lrc::clone(&cache_entry.file), cache_entry.line_number, col));
        }

        // No cache hit ...
        let oldest = self.oldest_cache_entry_index();

        // If the entry doesn't point to the correct file, get the new file and index.
        let new_file_and_idx = if !file_contains(&self.line_cache[oldest].file, pos) {
            Some(self.file_for_position(pos)?)
        } else {
            None
        };

        let cache_entry = &mut self.line_cache[oldest];
        cache_entry.update(new_file_and_idx, pos, self.time_stamp);

        let col = RelativeBytePos(pos.to_u32() - cache_entry.line.start.to_u32());
        Some((Lrc::clone(&cache_entry.file), cache_entry.line_number, col))
    }

    pub fn span_data_to_lines_and_cols(
        &mut self,
        span_data: &SpanData,
    ) -> Option<(StableSourceFileId, usize, BytePos, usize, BytePos)> {
        if self.source_map.files().is_empty() {
            return None;
        }

        let lo_idx = self.source_map.lookup_source_file_idx(span_data.lo);
        let hi_idx = self.source_map.lookup_source_file_idx(span_data.hi);

        if lo_idx != hi_idx {
            return None;
        }

        let file = &self.source_map.files()[lo_idx];

        let lincol = |absolute| {
            let relative = file.relative_position(absolute);
            let line = file.lookup_line(relative).unwrap();
            let bounds = file.line_bounds(line);
            (line + 1, absolute - bounds.start)
        };
        let (lo_line, lo_col) = lincol(span_data.lo);
        let (hi_line, hi_col) = lincol(span_data.hi);
        Some((file.stable_id, lo_line, lo_col, hi_line, hi_col))
    }

    fn cache_entry_index(&self, pos: BytePos) -> isize {
        for (idx, cache_entry) in self.line_cache.iter().enumerate() {
            if cache_entry.line.contains(&pos) {
                return idx as isize;
            }
        }

        -1
    }

    fn oldest_cache_entry_index(&self) -> usize {
        let mut oldest = 0;

        for idx in 1..self.line_cache.len() {
            if self.line_cache[idx].time_stamp < self.line_cache[oldest].time_stamp {
                oldest = idx;
            }
        }

        oldest
    }

    fn file_for_position(&self, pos: BytePos) -> Option<(Lrc<SourceFile>, usize)> {
        if !self.source_map.files().is_empty() {
            let file_idx = self.source_map.lookup_source_file_idx(pos);
            let file = &self.source_map.files()[file_idx];

            if file_contains(file, pos) {
                return Some((Lrc::clone(file), file_idx));
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
