// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
use fmt;
use iter::Cloned;
use slice::Iter;

/// This is just a table which allows mapping from a character to a string,
/// which at the moment is only used for `to_lowercase` and `to_uppercase`.
pub struct MappingTable {
    pub(crate) table: &'static [(char, [char; 3])],
}
impl MappingTable {
    pub fn lookup(&self, c: char) -> Lookup {
        let search = self.table.binary_search_by(|&(key, _)| key.cmp(&c)).ok();
        match search {
            None => Lookup(LookupInner::Same(c)),
            Some(index) => {
                let s = &self.table[index].1;
                match s.iter().position(|&c| c == '\0') {
                    None => Lookup(LookupInner::Iter(s.iter().cloned())),
                    Some(p) => Lookup(LookupInner::Iter(s[..p].iter().cloned())),
                }
            }
        }
    }
}

#[derive(Clone)]
pub enum LookupInner {
    Same(char),
    Iter(Cloned<Iter<'static, char>>),
}

/// Iterator over the characters in a mapping.
#[derive(Clone)]
pub struct Lookup(LookupInner);

impl Iterator for Lookup {
    type Item = char;

    #[inline]
    fn next(&mut self) -> Option<char> {
        let next;
        match &mut self.0 {
            LookupInner::Iter(iter) => return iter.next(),
            LookupInner::Same(c) => {
                next = *c;
            }
        }
        self.0 = LookupInner::Iter([].iter().cloned());
        Some(next)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match &self.0 {
            LookupInner::Same(_) => (1, Some(1)),
            LookupInner::Iter(iter) => iter.size_hint(),
        }
    }
}

impl fmt::Debug for Lookup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_list().entries(self.clone()).finish()
    }
}

impl fmt::Display for Lookup {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for c in self.clone() {
            fmt::Write::write_char(f, c)?;
        }
        Ok(())
    }
}
