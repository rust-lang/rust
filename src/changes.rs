// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// TODO
// print to files 
// tests

use strings::string_buffer::StringBuffer;
use std::collections::HashMap;
use syntax::codemap::{CodeMap, Span,BytePos};
use std::fmt;

// This is basically a wrapper around a bunch of Ropes which makes it convenient
// to work with libsyntax. It is badly named.
pub struct ChangeSet<'a> {
    file_map: HashMap<String, StringBuffer>,
    codemap: &'a CodeMap,
    file_spans: Vec<(u32, u32)>,
}

impl<'a> ChangeSet<'a> {
    // Create a new ChangeSet for a given libsyntax CodeMap.
    pub fn from_codemap(codemap: &'a CodeMap) -> ChangeSet<'a> {
        let mut result = ChangeSet {
            file_map: HashMap::new(),
            codemap: codemap,
            file_spans: Vec::with_capacity(codemap.files.borrow().len()),
        };

        for f in codemap.files.borrow().iter() {
            // Use the length of the file as a heuristic for how much space we
            // need. I hope that at some stage someone rounds this up to the next
            // power of two. TODO check that or do it here.
            result.file_map.insert(f.name.clone(),
                                   StringBuffer::with_capacity(f.src.as_ref().unwrap().len()));

            result.file_spans.push((f.start_pos.0, f.end_pos.0));
        }

        result.file_spans.sort();

        result
    }

    pub fn filespans_for_span(&self, start: BytePos, end: BytePos) -> Vec<(u32, u32)> {
        assert!(start.0 <= end.0);

        if self.file_spans.len() == 0 {
            return Vec::new();
        }

        let mut idx = match self.file_spans.binary_search(&(start.0, ::std::u32::MAX)) {
            Ok(i) => i,
            Err(0) => 0,
            Err(i) => i - 1,
        };

        let mut result = Vec::new();
        let mut start = start.0;
        loop {
            let cur_file = &self.file_spans[idx];
            idx += 1;

            if idx >= self.file_spans.len() || start >= end.0 {
                if start < end.0 {
                    result.push((start, end.0));
                }
                return result;
            }

            let end = ::std::cmp::min(cur_file.1 - 1, end.0);
            if start < end {
                result.push((start, end));
            }
            start = self.file_spans[idx].0;
        }
    }

    pub fn push_str(&mut self, file_name: &str, text: &str) {
        let buf = self.file_map.get_mut(&*file_name).unwrap();
        buf.push_str(text)
    }

    pub fn push_str_span(&mut self, span: Span, text: &str) {
        let file_name = self.codemap.span_to_filename(span);
        self.push_str(&file_name, text)
    }

    pub fn cur_offset(&mut self, file_name: &str) -> usize {
        self.file_map[&*file_name].cur_offset()
    }

    pub fn cur_offset_span(&mut self, span: Span) -> usize {
        let file_name = self.codemap.span_to_filename(span);
        self.cur_offset(&file_name)
    }

    // Return an iterator over the entire changed text.
    pub fn text<'c>(&'c self) -> FileIterator<'c, 'a> {
        FileIterator {
            change_set: self,
            keys: self.file_map.keys().collect(),
            cur_key: 0,
        }
    }

}

// Iterates over each file in the ChangSet. Yields the filename and the changed
// text for that file.
pub struct FileIterator<'c, 'a: 'c> {
    change_set: &'c ChangeSet<'a>,
    keys: Vec<&'c String>,
    cur_key: usize,
}

impl<'c, 'a> Iterator for FileIterator<'c, 'a> {
    type Item = (&'c str, &'c StringBuffer);

    fn next(&mut self) -> Option<(&'c str, &'c StringBuffer)> {
        if self.cur_key >= self.keys.len() {
            return None;
        }

        let key = self.keys[self.cur_key];
        self.cur_key += 1;
        return Some((&key, &self.change_set.file_map[&*key]))
    }
}

impl<'a> fmt::Display for ChangeSet<'a> {
    // Prints the entire changed text.
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for (f, r) in self.text() {
            try!(write!(fmt, "{}:\n", f));
            try!(write!(fmt, "{}\n\n", r));
        }
        Ok(())
    }    
}
