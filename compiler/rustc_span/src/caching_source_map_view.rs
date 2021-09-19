use crate::source_map::SourceMap;
use crate::{BytePos, SourceFile, SpanData};
use rustc_data_structures::sync::Lrc;
use std::ops::Range;

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
        let first_file = files[0].clone();
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
    ) -> Option<(Lrc<SourceFile>, usize, BytePos)> {
        self.time_stamp += 1;

        // Check if the position is in one of the cached lines
        let cache_idx = self.cache_entry_index(pos);
        if cache_idx != -1 {
            let cache_entry = &mut self.line_cache[cache_idx as usize];
            cache_entry.touch(self.time_stamp);

            return Some((
                cache_entry.file.clone(),
                cache_entry.line_number,
                pos - cache_entry.line.start,
            ));
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

        Some((cache_entry.file.clone(), cache_entry.line_number, pos - cache_entry.line.start))
    }

    pub fn span_data_to_lines_and_cols(
        &mut self,
        span_data: &SpanData,
    ) -> Option<(Lrc<SourceFile>, usize, BytePos, usize, BytePos)> {
        self.time_stamp += 1;

        // Check if lo and hi are in the cached lines.
        let lo_cache_idx = self.cache_entry_index(span_data.lo);
        let hi_cache_idx = self.cache_entry_index(span_data.hi);

        if lo_cache_idx != -1 && hi_cache_idx != -1 {
            // Cache hit for span lo and hi. Check if they belong to the same file.
            let result = {
                let lo = &self.line_cache[lo_cache_idx as usize];
                let hi = &self.line_cache[hi_cache_idx as usize];

                if lo.file_index != hi.file_index {
                    return None;
                }

                (
                    lo.file.clone(),
                    lo.line_number,
                    span_data.lo - lo.line.start,
                    hi.line_number,
                    span_data.hi - hi.line.start,
                )
            };

            self.line_cache[lo_cache_idx as usize].touch(self.time_stamp);
            self.line_cache[hi_cache_idx as usize].touch(self.time_stamp);

            return Some(result);
        }

        // No cache hit or cache hit for only one of span lo and hi.
        let oldest = if lo_cache_idx != -1 || hi_cache_idx != -1 {
            let avoid_idx = if lo_cache_idx != -1 { lo_cache_idx } else { hi_cache_idx };
            self.oldest_cache_entry_index_avoid(avoid_idx as usize)
        } else {
            self.oldest_cache_entry_index()
        };

        // If the entry doesn't point to the correct file, get the new file and index.
        // Return early if the file containing beginning of span doesn't contain end of span.
        let new_file_and_idx = if !file_contains(&self.line_cache[oldest].file, span_data.lo) {
            let new_file_and_idx = self.file_for_position(span_data.lo)?;
            if !file_contains(&new_file_and_idx.0, span_data.hi) {
                return None;
            }

            Some(new_file_and_idx)
        } else {
            let file = &self.line_cache[oldest].file;
            if !file_contains(&file, span_data.hi) {
                return None;
            }

            None
        };

        // Update the cache entries.
        let (lo_idx, hi_idx) = match (lo_cache_idx, hi_cache_idx) {
            // Oldest cache entry is for span_data.lo line.
            (-1, -1) => {
                let lo = &mut self.line_cache[oldest];
                lo.update(new_file_and_idx, span_data.lo, self.time_stamp);

                if !lo.line.contains(&span_data.hi) {
                    let new_file_and_idx = Some((lo.file.clone(), lo.file_index));
                    let next_oldest = self.oldest_cache_entry_index_avoid(oldest);
                    let hi = &mut self.line_cache[next_oldest];
                    hi.update(new_file_and_idx, span_data.hi, self.time_stamp);
                    (oldest, next_oldest)
                } else {
                    (oldest, oldest)
                }
            }
            // Oldest cache entry is for span_data.lo line.
            (-1, _) => {
                let lo = &mut self.line_cache[oldest];
                lo.update(new_file_and_idx, span_data.lo, self.time_stamp);
                let hi = &mut self.line_cache[hi_cache_idx as usize];
                hi.touch(self.time_stamp);
                (oldest, hi_cache_idx as usize)
            }
            // Oldest cache entry is for span_data.hi line.
            (_, -1) => {
                let hi = &mut self.line_cache[oldest];
                hi.update(new_file_and_idx, span_data.hi, self.time_stamp);
                let lo = &mut self.line_cache[lo_cache_idx as usize];
                lo.touch(self.time_stamp);
                (lo_cache_idx as usize, oldest)
            }
            _ => {
                panic!();
            }
        };

        let lo = &self.line_cache[lo_idx];
        let hi = &self.line_cache[hi_idx];

        // Span lo and hi may equal line end when last line doesn't
        // end in newline, hence the inclusive upper bounds below.
        assert!(span_data.lo >= lo.line.start);
        assert!(span_data.lo <= lo.line.end);
        assert!(span_data.hi >= hi.line.start);
        assert!(span_data.hi <= hi.line.end);
        assert!(lo.file.contains(span_data.lo));
        assert!(lo.file.contains(span_data.hi));
        assert_eq!(lo.file_index, hi.file_index);

        Some((
            lo.file.clone(),
            lo.line_number,
            span_data.lo - lo.line.start,
            hi.line_number,
            span_data.hi - hi.line.start,
        ))
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

    fn oldest_cache_entry_index_avoid(&self, avoid_idx: usize) -> usize {
        let mut oldest = if avoid_idx != 0 { 0 } else { 1 };

        for idx in 0..self.line_cache.len() {
            if idx != avoid_idx
                && self.line_cache[idx].time_stamp < self.line_cache[oldest].time_stamp
            {
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
                return Some((file.clone(), file_idx));
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
