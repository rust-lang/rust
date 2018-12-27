use rustc_data_structures::sync::Lrc;
use syntax::source_map::SourceMap;
use syntax_pos::{BytePos, SourceFile};

#[derive(Clone)]
struct CacheEntry {
    time_stamp: usize,
    line_number: usize,
    line_start: BytePos,
    line_end: BytePos,
    file: Lrc<SourceFile>,
    file_index: usize,
}

#[derive(Clone)]
pub struct CachingSourceMapView<'cm> {
    source_map: &'cm SourceMap,
    line_cache: [CacheEntry; 3],
    time_stamp: usize,
}

impl<'cm> CachingSourceMapView<'cm> {
    pub fn new(source_map: &'cm SourceMap) -> CachingSourceMapView<'cm> {
        let files = source_map.files();
        let first_file = files[0].clone();
        let entry = CacheEntry {
            time_stamp: 0,
            line_number: 0,
            line_start: BytePos(0),
            line_end: BytePos(0),
            file: first_file,
            file_index: 0,
        };

        CachingSourceMapView {
            source_map,
            line_cache: [entry.clone(), entry.clone(), entry],
            time_stamp: 0,
        }
    }

    pub fn byte_pos_to_line_and_col(&mut self,
                                    pos: BytePos)
                                    -> Option<(Lrc<SourceFile>, usize, BytePos)> {
        self.time_stamp += 1;

        // Check if the position is in one of the cached lines
        for cache_entry in self.line_cache.iter_mut() {
            if pos >= cache_entry.line_start && pos < cache_entry.line_end {
                cache_entry.time_stamp = self.time_stamp;

                return Some((cache_entry.file.clone(),
                             cache_entry.line_number,
                             pos - cache_entry.line_start));
            }
        }

        // No cache hit ...
        let mut oldest = 0;
        for index in 1 .. self.line_cache.len() {
            if self.line_cache[index].time_stamp < self.line_cache[oldest].time_stamp {
                oldest = index;
            }
        }

        let cache_entry = &mut self.line_cache[oldest];

        // If the entry doesn't point to the correct file, fix it up
        if pos < cache_entry.file.start_pos || pos >= cache_entry.file.end_pos {
            let file_valid;
            if self.source_map.files().len() > 0 {
                let file_index = self.source_map.lookup_source_file_idx(pos);
                let file = self.source_map.files()[file_index].clone();

                if pos >= file.start_pos && pos < file.end_pos {
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
        cache_entry.line_start = line_bounds.0;
        cache_entry.line_end = line_bounds.1;
        cache_entry.time_stamp = self.time_stamp;

        return Some((cache_entry.file.clone(),
                     cache_entry.line_number,
                     pos - cache_entry.line_start));
    }
}
