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
// composable changes
// print to files (maybe that shouldn't be here, but in mod)
// tests
// docs

use rope::{Rope, RopeSlice};
use std::collections::{HashMap, BTreeMap};
use std::collections::Bound::{Included, Unbounded};
use syntax::codemap::{CodeMap, Span, Pos};
use std::fmt;

pub struct ChangeSet<'a> {
    file_map: HashMap<String, Rope>,
    // FIXME, we only keep a codemap around so we can have convenience methods
    // taking Spans, it would be more resuable to factor this (and the methods)
    // out into an adaptor.
    codemap: &'a CodeMap,
    pub count: u64,
    // TODO we need to map the start and end of spans differently
    // TODO needs to be per file
    adjusts: BTreeMap<usize, Adjustment>,
}

// An extent over which we must adjust the position values.
#[derive(Show, Clone, Eq, PartialEq)]
struct Adjustment {
    // Start is implicit, given by its position in the map.
    end: usize,
    delta: isize,
}

impl Adjustment {
    fn chop_left(&self, new_end: usize) -> Adjustment {
        Adjustment {
            end: new_end,
            delta: self.delta,
        }
    }

    fn move_left(&self, mov: usize) -> Adjustment {
        assert!(self.delta > mov);
        Adjustment {
            end: self.end,
            delta: self.delta - mov,
        }
    }
}

pub struct FileIterator<'c, 'a: 'c> {
    change_set: &'c ChangeSet<'a>,
    keys: Vec<&'c String>,
    cur_key: usize,
}

impl<'a> ChangeSet<'a> {
    pub fn from_codemap(codemap: &'a CodeMap) -> ChangeSet<'a> {
        let mut result = ChangeSet {
            file_map: HashMap::new(),
            codemap: codemap,
            count: 0,
            adjusts: BTreeMap::new(),
        };

        for f in codemap.files.borrow().iter() {
            let contents = Rope::from_string(f.src.clone());
            result.file_map.insert(f.name.clone(), contents);
        }

        result
    }

    // start and end are unadjusted.
    pub fn change(&mut self, file_name: &str, start: usize, end: usize, text: String) {
        println!("change: {}:{}-{} \"{}\"", file_name, start, end, text);

        let new_len = text.len();
        self.count += 1;

        let (key_start, adj_start, abs_start): (Option<usize>, Option<Adjustment>, usize) = {
            let before_start = self.adjusts.range(Unbounded, Included(&start)).next_back();
            match before_start {
                Some((k, a)) if a.end > start => (Some(*k), Some(a.clone()), (start as isize + a.delta) as usize),
                _ => (None, None, start)
            }
        };
        let (key_end, adj_end, abs_end) = {
            let before_end = self.adjusts.range(Unbounded, Included(&end)).next_back();
            match before_end {
                Some((k, a)) if a.end > end => (Some(*k), Some(a.clone()), (end as isize + a.delta) as usize),
                _ => (None, None, end)
            }
        };

        {
            let file = &mut self.file_map[*file_name];

            println!("change: absolute values {}-{}, replaces \"{}\"",
                   abs_start, abs_end, file.slice(abs_start..abs_end));

            file.remove(abs_start, abs_end);
            file.insert(abs_start, text);

            // Record the changed locations.
            // TODO what if there is a change to the right of end? - need to iterate over all changes to the right :-(
            match (key_start, key_end) {
                (None, None) => {
                    // Factor this out?
                    let old_len = end as isize - start as isize;
                    let delta = new_len as isize - old_len;
                    self.adjusts.insert(end, Adjustment { end: file.len(), delta: delta });
                }
                (Some(k), None) => {
                    // Adjust the old change.
                    self.adjusts[k] = adj_start.unwrap().chop_left(end);

                    // Add the new one.
                    let old_len = end as isize - start as isize;
                    let delta = new_len as isize - old_len;
                    self.adjusts.insert(end, Adjustment { end: file.len(), delta: delta });
                }
                (None, Some(k)) => {
                    let old_len = end as isize - start as isize;
                    let delta = new_len as isize - old_len;

                    // Adjust the old change.
                    // TODO only if we move left, but what if moving right?
                    self.adjusts[abs_end] = adj_end.unwrap().move_left(TODO);
                    self.adjusts.remove(&k);

                    // Add the new one.
                    self.adjusts.insert(end, Adjustment { end: file.len(), delta: delta });
                }
                _ => {
                    println!("{}", file);
                    panic!();
                }
            }
        }

        debug_assert!(self.verify_adjustments(), "Bad change, created an overlapping adjustment");
    }

    // Intended for debugging.
    fn verify_adjustments(&self) -> bool {
        let mut prev_end = 0;
        let mut prev_delta = 0;
        for (&k, a) in self.adjusts.iter() {
            if k < prev_end {
                debug!("Found bad adjustment at start {}, overlaps with previous adjustment", k);
                return false;
            }
            if k as isize + a.delta < 0 {
                debug!("Found bad adjustment at start {}, absolute start < 0", k);
                return false;
            }
            if k as isize + a.delta < prev_end as isize + prev_delta {
                debug!("Found bad adjustment at start {}, \
                        projection overlaps with previous projection", k);
                return false;
            }
            // TODO Check end + delta <= file.len - needs per file

            prev_end = a.end;
            prev_delta = a.delta;
        }
        true
    }

    // span is unadjusted.
    pub fn change_span(&mut self, span: Span, text: String) {
        let l_loc = self.codemap.lookup_char_pos(span.lo);
        let file_offset = l_loc.file.start_pos.0;
        self.change(&l_loc.file.name[],
                    (span.lo.0 - file_offset) as usize,
                    (span.hi.0 - file_offset) as usize,
                    text)
    }

    // start and end are unadjusted.
    pub fn slice(&self, file_name: &str, start: usize, end: usize) -> RopeSlice {
        // TODO refactor with change?
        let abs_start = {
            let before_start = self.adjusts.range(Unbounded, Included(&start)).next_back();
            match before_start {
                Some((k, ref a)) if a.end > start => (start as isize + a.delta) as usize,
                _ => start
            }
        };
        let abs_end = {
            let before_end = self.adjusts.range(Unbounded, Included(&end)).next_back();
            match before_end {
                Some((k, ref a)) if a.end > end => (end as isize + a.delta) as usize,
                _ => end
            }
        };

        let file = &self.file_map[*file_name];
        file.slice(abs_start..abs_end)
    }

    // span is unadjusted.
    pub fn slice_span(&self, span:Span) -> RopeSlice {
        let l_loc = self.codemap.lookup_char_pos(span.lo);
        let file_offset = l_loc.file.start_pos.0;
        self.slice(&l_loc.file.name[],
                   (span.lo.0 - file_offset) as usize,
                   (span.hi.0 - file_offset) as usize)
    }

    pub fn text<'c>(&'c self) -> FileIterator<'c, 'a> {
        FileIterator {
            change_set: self,
            keys: self.file_map.keys().collect(),
            cur_key: 0,
        }
    }
}

impl<'c, 'a> Iterator for FileIterator<'c, 'a> {
    type Item = (&'c str, &'c Rope);
    fn next(&mut self) -> Option<(&'c str, &'c Rope)> {
        if self.cur_key >= self.keys.len() {
            return None;
        }

        let key = self.keys[self.cur_key];
        self.cur_key += 1;
        return Some((&key[], &self.change_set.file_map[*key]))
    }
}

impl<'a> fmt::Display for ChangeSet<'a> {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        for (f, r) in self.text() {
            try!(write!(fmt, "{}:\n", f));
            try!(write!(fmt, "{}", r));
        }
        Ok(())
    }    
}
