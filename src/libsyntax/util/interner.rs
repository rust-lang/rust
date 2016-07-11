// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! An "interner" is a data structure that associates values with usize tags and
//! allows bidirectional lookup; i.e. given a value, one can easily find the
//! type, and vice versa.

use ast::Name;

use std::borrow::Borrow;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(PartialEq, Eq, Hash)]
struct RcStr(Rc<String>);

impl Borrow<str> for RcStr {
    fn borrow(&self) -> &str {
        &self.0
    }
}

pub struct Interner {
    map: RefCell<HashMap<RcStr, Name>>,
    vect: RefCell<Vec<Rc<String>> >,
}

/// When traits can extend traits, we should extend index<Name,T> to get []
impl Interner {
    pub fn new() -> Self {
        Interner {
            map: RefCell::new(HashMap::new()),
            vect: RefCell::new(Vec::new()),
        }
    }

    pub fn prefill(init: &[&str]) -> Self {
        let rv = Interner::new();
        for &v in init { rv.intern(v); }
        rv
    }

    pub fn intern<T: Borrow<str> + Into<String>>(&self, val: T) -> Name {
        let mut map = self.map.borrow_mut();
        if let Some(&idx) = map.get(val.borrow()) {
            return idx;
        }

        let new_idx = Name(self.len() as u32);
        let val = Rc::new(val.into());
        map.insert(RcStr(val.clone()), new_idx);
        self.vect.borrow_mut().push(val);
        new_idx
    }

    pub fn gensym(&self, val: &str) -> Name {
        let new_idx = Name(self.len() as u32);
        // leave out of .map to avoid colliding
        self.vect.borrow_mut().push(Rc::new(val.to_owned()));
        new_idx
    }

    // I want these gensyms to share name pointers
    // with existing entries. This would be automatic,
    // except that the existing gensym creates its
    // own managed ptr using to_managed. I think that
    // adding this utility function is the most
    // lightweight way to get what I want, though not
    // necessarily the cleanest.

    /// Create a gensym with the same name as an existing
    /// entry.
    pub fn gensym_copy(&self, idx : Name) -> Name {
        let new_idx = Name(self.len() as u32);
        // leave out of map to avoid colliding
        let mut vect = self.vect.borrow_mut();
        let existing = (*vect)[idx.0 as usize].clone();
        vect.push(existing);
        new_idx
    }

    pub fn get(&self, idx: Name) -> Rc<String> {
        (*self.vect.borrow())[idx.0 as usize].clone()
    }

    pub fn len(&self) -> usize {
        self.vect.borrow().len()
    }

    pub fn find(&self, val: &str) -> Option<Name> {
        self.map.borrow().get(val).cloned()
    }

    pub fn clear(&self) {
        *self.map.borrow_mut() = HashMap::new();
        *self.vect.borrow_mut() = Vec::new();
    }

    pub fn reset(&self, other: Interner) {
        *self.map.borrow_mut() = other.map.into_inner();
        *self.vect.borrow_mut() = other.vect.into_inner();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ast::Name;

    #[test]
    fn interner_tests() {
        let i : Interner = Interner::new();
        // first one is zero:
        assert_eq!(i.intern("dog"), Name(0));
        // re-use gets the same entry:
        assert_eq!(i.intern ("dog"), Name(0));
        // different string gets a different #:
        assert_eq!(i.intern("cat"), Name(1));
        assert_eq!(i.intern("cat"), Name(1));
        // dog is still at zero
        assert_eq!(i.intern("dog"), Name(0));
        // gensym gets 3
        assert_eq!(i.gensym("zebra"), Name(2));
        // gensym of same string gets new number :
        assert_eq!(i.gensym("zebra"), Name(3));
        // gensym of *existing* string gets new number:
        assert_eq!(i.gensym("dog"), Name(4));
        // gensym tests again with gensym_copy:
        assert_eq!(i.gensym_copy(Name(2)), Name(5));
        assert_eq!(*i.get(Name(5)), "zebra");
        assert_eq!(i.gensym_copy(Name(2)), Name(6));
        assert_eq!(*i.get(Name(6)), "zebra");
        assert_eq!(*i.get(Name(0)), "dog");
        assert_eq!(*i.get(Name(1)), "cat");
        assert_eq!(*i.get(Name(2)), "zebra");
        assert_eq!(*i.get(Name(3)), "zebra");
        assert_eq!(*i.get(Name(4)), "dog");
    }
}
