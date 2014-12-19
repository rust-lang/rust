// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use prelude::*;
use std::rand;
use std::rand::Rng;
use test::Bencher;

pub fn insert_rand_n<M, I, R>(n: uint,
                              map: &mut M,
                              b: &mut Bencher,
                              mut insert: I,
                              mut remove: R) where
    I: FnMut(&mut M, uint),
    R: FnMut(&mut M, uint),
{
    // setup
    let mut rng = rand::weak_rng();

    for _ in range(0, n) {
        insert(map, rng.gen::<uint>() % n);
    }

    // measure
    b.iter(|| {
        let k = rng.gen::<uint>() % n;
        insert(map, k);
        remove(map, k);
    })
}

pub fn insert_seq_n<M, I, R>(n: uint,
                             map: &mut M,
                             b: &mut Bencher,
                             mut insert: I,
                             mut remove: R) where
    I: FnMut(&mut M, uint),
    R: FnMut(&mut M, uint),
{
    // setup
    for i in range(0u, n) {
        insert(map, i * 2);
    }

    // measure
    let mut i = 1;
    b.iter(|| {
        insert(map, i);
        remove(map, i);
        i = (i + 2) % n;
    })
}

pub fn find_rand_n<M, T, I, F>(n: uint,
                               map: &mut M,
                               b: &mut Bencher,
                               mut insert: I,
                               mut find: F) where
    I: FnMut(&mut M, uint),
    F: FnMut(&M, uint) -> T,
{
    // setup
    let mut rng = rand::weak_rng();
    let mut keys = Vec::from_fn(n, |_| rng.gen::<uint>() % n);

    for k in keys.iter() {
        insert(map, *k);
    }

    rng.shuffle(keys.as_mut_slice());

    // measure
    let mut i = 0;
    b.iter(|| {
        let t = find(map, keys[i]);
        i = (i + 1) % n;
        t
    })
}

pub fn find_seq_n<M, T, I, F>(n: uint,
                              map: &mut M,
                              b: &mut Bencher,
                              mut insert: I,
                              mut find: F) where
    I: FnMut(&mut M, uint),
    F: FnMut(&M, uint) -> T,
{
    // setup
    for i in range(0u, n) {
        insert(map, i);
    }

    // measure
    let mut i = 0;
    b.iter(|| {
        let x = find(map, i);
        i = (i + 1) % n;
        x
    })
}
