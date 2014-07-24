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

#[cfg(test)]
pub mod bench {
    use std::prelude::*;
    use std::rand;
    use std::rand::Rng;
    use test::Bencher;
    use MutableMap;

    pub fn insert_rand_n<M:MutableMap<uint,uint>>(n: uint,
                                                  map: &mut M,
                                                  b: &mut Bencher) {
        // setup
        let mut rng = rand::weak_rng();

        map.clear();
        for _ in range(0, n) {
            map.insert(rng.gen::<uint>() % n, 1);
        }

        // measure
        b.iter(|| {
            let k = rng.gen::<uint>() % n;
            map.insert(k, 1);
            map.remove(&k);
        })
    }

    pub fn insert_seq_n<M:MutableMap<uint,uint>>(n: uint,
                                                 map: &mut M,
                                                 b: &mut Bencher) {
        // setup
        map.clear();
        for i in range(0u, n) {
            map.insert(i*2, 1);
        }

        // measure
        let mut i = 1;
        b.iter(|| {
            map.insert(i, 1);
            map.remove(&i);
            i = (i + 2) % n;
        })
    }

    pub fn find_rand_n<M:MutableMap<uint,uint>>(n: uint,
                                                map: &mut M,
                                                b: &mut Bencher) {
        // setup
        let mut rng = rand::weak_rng();
        let mut keys = Vec::from_fn(n, |_| rng.gen::<uint>() % n);

        for k in keys.iter() {
            map.insert(*k, 1);
        }

        rng.shuffle(keys.as_mut_slice());

        // measure
        let mut i = 0;
        b.iter(|| {
            map.find(&keys[i]);
            i = (i + 1) % n;
        })
    }

    pub fn find_seq_n<M:MutableMap<uint,uint>>(n: uint,
                                               map: &mut M,
                                               b: &mut Bencher) {
        // setup
        for i in range(0u, n) {
            map.insert(i, 1);
        }

        // measure
        let mut i = 0;
        b.iter(|| {
            let x = map.find(&i);
            i = (i + 1) % n;
            x
        })
     }
}

