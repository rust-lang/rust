use crate::source_map::SourceMap;
use crate::{BytePos, SourceFile};
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
        for cache_entry in self.line_cache.iter_mut() {
            if cache_entry.line.contains(&pos) {
                cache_entry.time_stamp = self.time_stamp;

                return Some((
                    cache_entry.file.clone(),
                    cache_entry.line_number,
                    pos - cache_entry.line.start,
                ));
            }
        }

        // No cache hit ...
        let mut oldest = 0;
        for index in 1..self.line_cache.len() {
            if self.line_cache[index].time_stamp < self.line_cache[oldest].time_stamp {
                oldest = index;
            }
        }

        let cache_entry = &mut self.line_cache[oldest];

        // If the entry doesn't point to the correct file, fix it up
        if !file_contains(&cache_entry.file, pos) {
            let file_valid;
            if self.source_map.files().len() > 0 {
                let file_index = self.source_map.lookup_source_file_idx(pos);
                let file = self.source_map.files()[file_index].clone();

                if file_contains(&file, pos) {
                    cache_entry.file = file;
                    cache_entry.file_index = file_index;
                    file_valid = true;
                } else {
                    file_valid = false;
                }
            } else {
                file_valid = false;
            }

            if !file_valid {
                return None;
            }
        }

        let line_index = cache_entry.file.lookup_line(pos).unwrap();
        let line_bounds = cache_entry.file.line_bounds(line_index);

        cache_entry.line_number = line_index + 1;
        cache_entry.line = line_bounds;
        cache_entry.time_stamp = self.time_stamp;

        Some((cache_entry.file.clone(), cache_entry.line_number, pos - cache_entry.line.start))
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
