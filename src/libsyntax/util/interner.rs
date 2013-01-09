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

use core::prelude::*;

use core::dvec::DVec;
use std::map::HashMap;
use std::map;

type hash_interner<T: Const> =
    {map: HashMap<T, uint>,
     vect: DVec<T>};

fn mk<T:Eq IterBytes Hash Const Copy>() -> Interner<T> {
    let m = map::HashMap::<T, uint>();
    let hi: hash_interner<T> =
        {map: m, vect: DVec()};
    move ((move hi) as Interner::<T>)
}

fn mk_prefill<T:Eq IterBytes Hash Const Copy>(init: ~[T]) -> Interner<T> {
    let rv = mk();
    for init.each() |v| { rv.intern(*v); }
    return rv;
}


/* when traits can extend traits, we should extend index<uint,T> to get [] */
trait Interner<T:Eq IterBytes Hash Const Copy> {
    fn intern(T) -> uint;
    fn gensym(T) -> uint;
    pure fn get(uint) -> T;
    fn len() -> uint;
}

impl <T:Eq IterBytes Hash Const Copy> hash_interner<T>: Interner<T> {
    fn intern(val: T) -> uint {
        match self.map.find(val) {
          Some(idx) => return idx,
          None => {
            let new_idx = self.vect.len();
            self.map.insert(val, new_idx);
            self.vect.push(val);
            return new_idx;
          }
        }
    }
    fn gensym(val: T) -> uint {
        let new_idx = self.vect.len();
        // leave out of .map to avoid colliding
        self.vect.push(val);
        return new_idx;
    }

    // this isn't "pure" in the traditional sense, because it can go from
    // failing to returning a value as items are interned. But for typestate,
    // where we first check a pred and then rely on it, ceasing to fail is ok.
    pure fn get(idx: uint) -> T { self.vect.get_elt(idx) }

    fn len() -> uint { return self.vect.len(); }
}
