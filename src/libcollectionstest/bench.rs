// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! map_insert_rand_bench {
    ($name: ident, $n: expr, $map: ident) => (
        #[bench]
        pub fn $name(b: &mut ::test::Bencher) {
            use std::rand;
            use std::rand::Rng;
            use test::black_box;

            let n: usize = $n;
            let mut map = $map::new();
            // setup
            let mut rng = rand::weak_rng();

            for _ in 0..n {
                let i = rng.gen() % n;
                map.insert(i, i);
            }

            // measure
            b.iter(|| {
                let k = rng.gen() % n;
                map.insert(k, k);
                map.remove(&k);
            });
            black_box(map);
        }
    )
}

macro_rules! map_insert_seq_bench {
    ($name: ident, $n: expr, $map: ident) => (
        #[bench]
        pub fn $name(b: &mut ::test::Bencher) {
            use test::black_box;

            let mut map = $map::new();
            let n: usize = $n;
            // setup
            for i in 0..n {
                map.insert(i * 2, i * 2);
            }

            // measure
            let mut i = 1;
            b.iter(|| {
                map.insert(i, i);
                map.remove(&i);
                i = (i + 2) % n;
            });
            black_box(map);
        }
    )
}

macro_rules! map_find_rand_bench {
    ($name: ident, $n: expr, $map: ident) => (
        #[bench]
        pub fn $name(b: &mut ::test::Bencher) {
            use std::iter::IteratorExt;
            use std::rand::Rng;
            use std::rand;
            use std::vec::Vec;
            use test::black_box;

            let mut map = $map::new();
            let n: usize = $n;

            // setup
            let mut rng = rand::weak_rng();
            let mut keys: Vec<_> = (0..n).map(|_| rng.gen() % n).collect();

            for &k in &keys {
                map.insert(k, k);
            }

            rng.shuffle(&mut keys);

            // measure
            let mut i = 0;
            b.iter(|| {
                let t = map.get(&keys[i]);
                i = (i + 1) % n;
                black_box(t);
            })
        }
    )
}

macro_rules! map_find_seq_bench {
    ($name: ident, $n: expr, $map: ident) => (
        #[bench]
        pub fn $name(b: &mut ::test::Bencher) {
            use test::black_box;

            let mut map = $map::new();
            let n: usize = $n;

            // setup
            for i in 0..n {
                map.insert(i, i);
            }

            // measure
            let mut i = 0;
            b.iter(|| {
                let x = map.get(&i);
                i = (i + 1) % n;
                black_box(x);
            })
        }
    )
}
