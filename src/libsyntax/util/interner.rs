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

// allow the interner_key macro to escape this module:
#[macro_escape];

use core::prelude::*;

use core::cmp::Equiv;
use core::hashmap::HashMap;
use syntax::parse::token::StringRef;

pub struct Interner<T> {
    priv map: @mut HashMap<T, uint>,
    priv vect: @mut ~[T],
}

// when traits can extend traits, we should extend index<uint,T> to get []
impl<T:Eq + IterBytes + Hash + Const + Copy> Interner<T> {
    pub fn new() -> Interner<T> {
        Interner {
            map: @mut HashMap::new(),
            vect: @mut ~[],
        }
    }

    pub fn prefill(init: &[T]) -> Interner<T> {
        let rv = Interner::new();
        for init.each() |v| { rv.intern(*v); }
        rv
    }

    pub fn intern(&self, val: T) -> uint {
        match self.map.find(&val) {
            Some(&idx) => return idx,
            None => (),
        }

        let vect = &mut *self.vect;
        let new_idx = vect.len();
        self.map.insert(val, new_idx);
        vect.push(val);
        new_idx
    }

    pub fn gensym(&self, val: T) -> uint {
        let new_idx = {
            let vect = &*self.vect;
            vect.len()
        };
        // leave out of .map to avoid colliding
        self.vect.push(val);
        new_idx
    }

    // this isn't "pure" in the traditional sense, because it can go from
    // failing to returning a value as items are interned. But for typestate,
    // where we first check a pred and then rely on it, ceasing to fail is ok.
    pub fn get(&self, idx: uint) -> T { self.vect[idx] }

    pub fn len(&self) -> uint { let vect = &*self.vect; vect.len() }

    pub fn find_equiv<Q:Hash + IterBytes + Equiv<T>>(&self, val: &Q)
                                              -> Option<uint> {
        match self.map.find_equiv(val) {
            Some(v) => Some(*v),
            None => None,
        }
    }
}

// A StrInterner differs from Interner<String> in that it accepts
// borrowed pointers rather than @ ones, resulting in less allocation.
pub struct StrInterner {
    priv map: @mut HashMap<@~str, uint>,
    priv vect: @mut ~[@~str],
}

// when traits can extend traits, we should extend index<uint,T> to get []
impl StrInterner {
    pub fn new() -> StrInterner {
        StrInterner {
            map: @mut HashMap::new(),
            vect: @mut ~[],
        }
    }

    pub fn prefill(init: &[&str]) -> StrInterner {
        let rv = StrInterner::new();
        for init.each() |v| { rv.intern(*v); }
        rv
    }

    pub fn intern(&self, val: &str) -> uint {
        match self.map.find_equiv(&StringRef(val)) {
            Some(&idx) => return idx,
            None => (),
        }

        let new_idx = self.len();
        self.map.insert(@val.to_owned(), new_idx);
        self.vect.push(@val.to_owned());
        new_idx
    }

    pub fn gensym(&self, val: &str) -> uint {
        let new_idx = self.len();
        // leave out of .map to avoid colliding
        self.vect.push(@val.to_owned());
        new_idx
    }

    // this isn't "pure" in the traditional sense, because it can go from
    // failing to returning a value as items are interned. But for typestate,
    // where we first check a pred and then rely on it, ceasing to fail is ok.
    pub fn get(&self, idx: uint) -> @~str { self.vect[idx] }

    pub fn len(&self) -> uint { let vect = &*self.vect; vect.len() }

    pub fn find_equiv<Q:Hash + IterBytes + Equiv<@~str>>(&self, val: &Q)
                                                         -> Option<uint> {
        match self.map.find_equiv(val) {
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
        let i : Interner<@~str> = Interner::new();
        i.get(13);
    }

    #[test]
    fn i2 () {
        let i : Interner<@~str> = Interner::new();
        // first one is zero:
        assert_eq!(i.intern (@~"dog"), 0);
        // re-use gets the same entry:
        assert_eq!(i.intern (@~"dog"), 0);
        // different string gets a different #:
        assert_eq!(i.intern (@~"cat"), 1);
        assert_eq!(i.intern (@~"cat"), 1);
        // dog is still at zero
        assert_eq!(i.intern (@~"dog"), 0);
        // gensym gets 3
        assert_eq!(i.gensym (@~"zebra" ), 2);
        // gensym of same string gets new number :
        assert_eq!(i.gensym (@~"zebra" ), 3);
        // gensym of *existing* string gets new number:
        assert_eq!(i.gensym (@~"dog"), 4);
        assert_eq!(i.get(0), @~"dog");
        assert_eq!(i.get(1), @~"cat");
        assert_eq!(i.get(2), @~"zebra");
        assert_eq!(i.get(3), @~"zebra");
        assert_eq!(i.get(4), @~"dog");
    }

    #[test]
    fn i3 () {
        let i : Interner<@~str> = Interner::prefill([@~"Alan",@~"Bob",@~"Carol"]);
        assert_eq!(i.get(0), @~"Alan");
        assert_eq!(i.get(1), @~"Bob");
        assert_eq!(i.get(2), @~"Carol");
        assert_eq!(i.intern(@~"Bob"), 1);
    }
}
