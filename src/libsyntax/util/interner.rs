// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// An "interner" is a data structure that associates values with uint tags and
// allows bidirectional lookup; i.e. given a value, one can easily find the
// type, and vice versa.

use ast::Name;

use std::cell::RefCell;
use std::cmp::Equiv;
use std::hashmap::HashMap;

pub struct Interner<T> {
    priv map: @RefCell<HashMap<T, Name>>,
    priv vect: @RefCell<~[T]>,
}

// when traits can extend traits, we should extend index<Name,T> to get []
impl<T:Eq + IterBytes + Hash + Freeze + Clone + 'static> Interner<T> {
    pub fn new() -> Interner<T> {
        Interner {
            map: @RefCell::new(HashMap::new()),
            vect: @RefCell::new(~[]),
        }
    }

    pub fn prefill(init: &[T]) -> Interner<T> {
        let rv = Interner::new();
        for v in init.iter() {
            rv.intern((*v).clone());
        }
        rv
    }

    pub fn intern(&self, val: T) -> Name {
        let mut map = self.map.borrow_mut();
        match map.get().find(&val) {
            Some(&idx) => return idx,
            None => (),
        }

        let mut vect = self.vect.borrow_mut();
        let new_idx = vect.get().len() as Name;
        map.get().insert(val.clone(), new_idx);
        vect.get().push(val);
        new_idx
    }

    pub fn gensym(&self, val: T) -> Name {
        let mut vect = self.vect.borrow_mut();
        let new_idx = vect.get().len() as Name;
        // leave out of .map to avoid colliding
        vect.get().push(val);
        new_idx
    }

    pub fn get(&self, idx: Name) -> T {
        let vect = self.vect.borrow();
        vect.get()[idx].clone()
    }

    pub fn len(&self) -> uint {
        let vect = self.vect.borrow();
        vect.get().len()
    }

    pub fn find_equiv<Q:Hash + IterBytes + Equiv<T>>(&self, val: &Q)
                                              -> Option<Name> {
        let map = self.map.borrow();
        match map.get().find_equiv(val) {
            Some(v) => Some(*v),
            None => None,
        }
    }
}

// A StrInterner differs from Interner<String> in that it accepts
// references rather than @ ones, resulting in less allocation.
pub struct StrInterner {
    priv map: @RefCell<HashMap<@str, Name>>,
    priv vect: @RefCell<~[@str]>,
}

// when traits can extend traits, we should extend index<Name,T> to get []
impl StrInterner {
    pub fn new() -> StrInterner {
        StrInterner {
            map: @RefCell::new(HashMap::new()),
            vect: @RefCell::new(~[]),
        }
    }

    pub fn prefill(init: &[&str]) -> StrInterner {
        let rv = StrInterner::new();
        for &v in init.iter() { rv.intern(v); }
        rv
    }

    pub fn intern(&self, val: &str) -> Name {
        let mut map = self.map.borrow_mut();
        match map.get().find_equiv(&val) {
            Some(&idx) => return idx,
            None => (),
        }

        let new_idx = self.len() as Name;
        let val = val.to_managed();
        map.get().insert(val, new_idx);
        let mut vect = self.vect.borrow_mut();
        vect.get().push(val);
        new_idx
    }

    pub fn gensym(&self, val: &str) -> Name {
        let new_idx = self.len() as Name;
        // leave out of .map to avoid colliding
        let mut vect = self.vect.borrow_mut();
        vect.get().push(val.to_managed());
        new_idx
    }

    // I want these gensyms to share name pointers
    // with existing entries. This would be automatic,
    // except that the existing gensym creates its
    // own managed ptr using to_managed. I think that
    // adding this utility function is the most
    // lightweight way to get what I want, though not
    // necessarily the cleanest.

    // create a gensym with the same name as an existing
    // entry.
    pub fn gensym_copy(&self, idx : Name) -> Name {
        let new_idx = self.len() as Name;
        // leave out of map to avoid colliding
        let mut vect = self.vect.borrow_mut();
        let existing = vect.get()[idx];
        vect.get().push(existing);
        new_idx
    }

    pub fn get(&self, idx: Name) -> @str {
        let vect = self.vect.borrow();
        vect.get()[idx]
    }

    pub fn len(&self) -> uint {
        let vect = self.vect.borrow();
        vect.get().len()
    }

    pub fn find_equiv<Q:Hash + IterBytes + Equiv<@str>>(&self, val: &Q)
                                                         -> Option<Name> {
        let map = self.map.borrow();
        match map.get().find_equiv(val) {
            Some(v) => Some(*v),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    #[should_fail]
    fn i1 () {
        let i : Interner<@str> = Interner::new();
        i.get(13);
    }

    #[test]
    fn interner_tests () {
        let i : Interner<@str> = Interner::new();
        // first one is zero:
        assert_eq!(i.intern(@"dog"), 0);
        // re-use gets the same entry:
        assert_eq!(i.intern(@"dog"), 0);
        // different string gets a different #:
        assert_eq!(i.intern(@"cat"), 1);
        assert_eq!(i.intern(@"cat"), 1);
        // dog is still at zero
        assert_eq!(i.intern(@"dog"), 0);
        // gensym gets 3
        assert_eq!(i.gensym(@"zebra" ), 2);
        // gensym of same string gets new number :
        assert_eq!(i.gensym (@"zebra" ), 3);
        // gensym of *existing* string gets new number:
        assert_eq!(i.gensym(@"dog"), 4);
        assert_eq!(i.get(0), @"dog");
        assert_eq!(i.get(1), @"cat");
        assert_eq!(i.get(2), @"zebra");
        assert_eq!(i.get(3), @"zebra");
        assert_eq!(i.get(4), @"dog");
    }

    #[test]
    fn i3 () {
        let i : Interner<@str> = Interner::prefill([@"Alan",@"Bob",@"Carol"]);
        assert_eq!(i.get(0), @"Alan");
        assert_eq!(i.get(1), @"Bob");
        assert_eq!(i.get(2), @"Carol");
        assert_eq!(i.intern(@"Bob"), 1);
    }

    #[test]
    fn string_interner_tests() {
        let i : StrInterner = StrInterner::new();
        // first one is zero:
        assert_eq!(i.intern("dog"), 0);
        // re-use gets the same entry:
        assert_eq!(i.intern ("dog"), 0);
        // different string gets a different #:
        assert_eq!(i.intern("cat"), 1);
        assert_eq!(i.intern("cat"), 1);
        // dog is still at zero
        assert_eq!(i.intern("dog"), 0);
        // gensym gets 3
        assert_eq!(i.gensym("zebra"), 2);
        // gensym of same string gets new number :
        assert_eq!(i.gensym("zebra"), 3);
        // gensym of *existing* string gets new number:
        assert_eq!(i.gensym("dog"), 4);
        // gensym tests again with gensym_copy:
        assert_eq!(i.gensym_copy(2), 5);
        assert_eq!(i.get(5), @"zebra");
        assert_eq!(i.gensym_copy(2), 6);
        assert_eq!(i.get(6), @"zebra");
        assert_eq!(i.get(0), @"dog");
        assert_eq!(i.get(1), @"cat");
        assert_eq!(i.get(2), @"zebra");
        assert_eq!(i.get(3), @"zebra");
        assert_eq!(i.get(4), @"dog");
    }
}
