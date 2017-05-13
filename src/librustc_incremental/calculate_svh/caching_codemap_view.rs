// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use rustc::ty::TyCtxt;
use std::rc::Rc;
use syntax::codemap::CodeMap;
use syntax_pos::{BytePos, FileMap};

#[derive(Clone)]
struct CacheEntry {
    time_stamp: usize,
    line_number: usize,
    line_start: BytePos,
    line_end: BytePos,
    file: Rc<FileMap>,
}

pub struct CachingCodemapView<'tcx> {
    codemap: &'tcx CodeMap,
    line_cache: [CacheEntry; 3],
    time_stamp: usize,
}

impl<'tcx> CachingCodemapView<'tcx> {
    pub fn new<'a>(tcx: TyCtxt<'a, 'tcx, 'tcx>) -> CachingCodemapView<'tcx> {
        let codemap = tcx.sess.codemap();
        let first_file = codemap.files.borrow()[0].clone();
        let entry = CacheEntry {
            time_stamp: 0,
            line_number: 0,
            line_start: BytePos(0),
            line_end: BytePos(0),
            file: first_file,
        };

        CachingCodemapView {
            codemap: codemap,
            line_cache: [entry.clone(), entry.clone(), entry.clone()],
            time_stamp: 0,
        }
    }

    pub fn codemap(&self) -> &'tcx CodeMap {
        self.codemap
    }

    pub fn byte_pos_to_line_and_col(&mut self,
                                    pos: BytePos)
                                    -> Option<(Rc<FileMap>, usize, BytePos)> {
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
            let files = self.codemap.files.borrow();

            if files.len() > 0 {
                let file_index = self.codemap.lookup_filemap_idx(pos);
                let file = files[file_index].clone();

                if pos >= file.start_pos && pos < file.end_pos {
                    cache_entry.file = file;
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
