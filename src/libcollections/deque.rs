// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

//! Container traits for collections

use std::container::Mutable;

/// A double-ended sequence that allows querying, insertion and deletion at both ends.
pub trait Deque<T> : Mutable {
    /// Provide a reference to the front element, or None if the sequence is empty
    fn front<'a>(&'a self) -> Option<&'a T>;

    /// Provide a mutable reference to the front element, or None if the sequence is empty
    fn front_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Provide a reference to the back element, or None if the sequence is empty
    fn back<'a>(&'a self) -> Option<&'a T>;

    /// Provide a mutable reference to the back element, or None if the sequence is empty
    fn back_mut<'a>(&'a mut self) -> Option<&'a mut T>;

    /// Insert an element first in the sequence
    fn push_front(&mut self, elt: T);

    /// Insert an element last in the sequence
    fn push_back(&mut self, elt: T);

    /// Remove the last element and return it, or None if the sequence is empty
    fn pop_back(&mut self) -> Option<T>;

    /// Remove the first element and return it, or None if the sequence is empty
    fn pop_front(&mut self) -> Option<T>;
}

#[cfg(test)]
pub mod bench {
    extern crate test;
    use self::test::BenchHarness;
    use std::container::MutableMap;
    use rand;
    use rand::Rng;

    pub fn insert_rand_n<M:MutableMap<uint,uint>>(n: uint,
                                                  map: &mut M,
                                                  bh: &mut BenchHarness) {
        // setup
        let mut rng = rand::weak_rng();

        map.clear();
        for _ in range(0, n) {
            map.insert(rng.gen::<uint>() % n, 1);
        }

        // measure
        bh.iter(|| {
            let k = rng.gen::<uint>() % n;
            map.insert(k, 1);
            map.remove(&k);
        })
    }

    pub fn insert_seq_n<M:MutableMap<uint,uint>>(n: uint,
                                                 map: &mut M,
                                                 bh: &mut BenchHarness) {
        // setup
        map.clear();
        for i in range(0u, n) {
            map.insert(i*2, 1);
        }

        // measure
        let mut i = 1;
        bh.iter(|| {
            map.insert(i, 1);
            map.remove(&i);
            i = (i + 2) % n;
        })
    }

    pub fn find_rand_n<M:MutableMap<uint,uint>>(n: uint,
                                                map: &mut M,
                                                bh: &mut BenchHarness) {
        // setup
        let mut rng = rand::weak_rng();
        let mut keys = Vec::from_fn(n, |_| rng.gen::<uint>() % n);

        for k in keys.iter() {
            map.insert(*k, 1);
        }

        rng.shuffle(keys.as_mut_slice());

        // measure
        let mut i = 0;
        bh.iter(|| {
            map.find(keys.get(i));
            i = (i + 1) % n;
        })
    }

    pub fn find_seq_n<M:MutableMap<uint,uint>>(n: uint,
                                               map: &mut M,
                                               bh: &mut BenchHarness) {
        // setup
        for i in range(0u, n) {
            map.insert(i, 1);
        }

        // measure
        let mut i = 0;
        bh.iter(|| {
            let x = map.find(&i);
            i = (i + 1) % n;
            x
        })
     }
}
