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
// print to files (maybe that shouldn't be here, but in mod)
// tests

use rope::{Rope, RopeSlice};
use std::collections::HashMap;
use syntax::codemap::{CodeMap, Span, BytePos};
use std::fmt;

// This is basically a wrapper around a bunch of Ropes which makes it convenient
// to work with libsyntax. It is badly named.
pub struct ChangeSet<'a> {
    file_map: HashMap<String, Rope>,
    codemap: &'a CodeMap,
    pub count: u64,
}

impl<'a> ChangeSet<'a> {
    // Create a new ChangeSet for a given libsyntax CodeMap.
    pub fn from_codemap(codemap: &'a CodeMap) -> ChangeSet<'a> {
        let mut result = ChangeSet {
            file_map: HashMap::new(),
            codemap: codemap,
            count: 0,
        };

        for f in codemap.files.borrow().iter() {
            let contents = Rope::from_string((&**f.src.as_ref().unwrap()).clone());
            result.file_map.insert(f.name.clone(), contents);
        }

        result
    }

    // Change a span of text in our stored text into the new text (`text`).
    // The span of text to change is given in the coordinates of the original
    // source text, not the current text,
    pub fn change(&mut self, file_name: &str, start: usize, end: usize, text: String) {
        println!("change: {}:{}-{} \"{}\"", file_name, start, end, text);

        self.count += 1;

        let file = &mut self.file_map[*file_name];

        if end - start == text.len() {
            // TODO src_replace_str would be much more efficient
            //file.src_replace_str(start, &text);
            file.src_remove(start, end);
            file.src_insert(start, text);
        } else {
            // TODO if we do this in one op, could we get better change info?
            file.src_remove(start, end);
            file.src_insert(start, text);
        }
    }

    // As for `change()`, but use a Span to indicate the text to change.
    pub fn change_span(&mut self, span: Span, text: String) {
        let l_loc = self.codemap.lookup_char_pos(span.lo);
        let file_offset = l_loc.file.start_pos.0;
        self.change(&l_loc.file.name,
                    (span.lo.0 - file_offset) as usize,
                    (span.hi.0 - file_offset) as usize,
                    text)
    }

    // Get a slice of the current text. Coordinates are relative to the source
    // text. I.e., this method returns the text which has been changed from the
    // indicated span.
    pub fn slice(&self, file_name: &str, start: usize, end: usize) -> RopeSlice {
        let file = &self.file_map[*file_name];
        file.src_slice(start..end)
    }

    // As for `slice()`, but use a Span to indicate the text to return.
    pub fn slice_span(&self, span:Span) -> RopeSlice {
        let l_loc = self.codemap.lookup_char_pos(span.lo);
        let file_offset = l_loc.file.start_pos.0;
        self.slice(&l_loc.file.name,
                   (span.lo.0 - file_offset) as usize,
                   (span.hi.0 - file_offset) as usize)
    }

    // Return an iterator over the entire changed text.
    pub fn text<'c>(&'c self) -> FileIterator<'c, 'a> {
        FileIterator {
            change_set: self,
            keys: self.file_map.keys().collect(),
            cur_key: 0,
        }
    }

    // Get the current line-relative position of a position in the source text.
    pub fn col(&self, loc: BytePos) -> usize {
        let l_loc = self.codemap.lookup_char_pos(loc);
        let file_offset = l_loc.file.start_pos.0;
        let file = &self.file_map[l_loc.file.name[..]];
        file.col_for_src_loc(loc.0 as usize - file_offset as usize)
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
    type Item = (&'c str, &'c Rope);

    fn next(&mut self) -> Option<(&'c str, &'c Rope)> {
        if self.cur_key >= self.keys.len() {
            return None;
        }

        let key = self.keys[self.cur_key];
        self.cur_key += 1;
        return Some((&key, &self.change_set.file_map[*key]))
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
