// This benchmark suite contains some benchmarks along a set of dimensions:
//   Hasher: std default (SipHash) and crate default (FxHash).
//   Int key distribution: low bit heavy, top bit heavy, and random.
//   Task: basic functionality: insert, insert_erase, lookup, lookup_fail, iter
#![feature(test)]

extern crate test;

use test::{black_box, Bencher};

use hashbrown::hash_map::DefaultHashBuilder;
use hashbrown::HashMap;
use std::collections::hash_map::RandomState;

const SIZE: usize = 1000;

// The default hashmap when using this crate directly.
type FxHashMap<K, V> = HashMap<K, V, DefaultHashBuilder>;
// This uses the hashmap from this crate with the default hasher of the stdlib.
type StdHashMap<K, V> = HashMap<K, V, RandomState>;

// A random key iterator.
#[derive(Clone, Copy)]
struct RandomKeys {
    state: usize,
}

impl RandomKeys {
    fn new() -> Self {
        RandomKeys { state: 0 }
    }
}

impl Iterator for RandomKeys {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        // Add 1 then multiply by some 32 bit prime.
        self.state = self.state.wrapping_add(1).wrapping_mul(3787392781);
        Some(self.state)
    }
}

macro_rules! bench_suite {
    ($bench_macro:ident, $bench_fx_serial:ident, $bench_std_serial:ident,
     $bench_fx_highbits:ident, $bench_std_highbits:ident,
     $bench_fx_random:ident, $bench_std_random:ident) => {
        $bench_macro!($bench_fx_serial, FxHashMap, 0..);
        $bench_macro!($bench_std_serial, StdHashMap, 0..);
        $bench_macro!($bench_fx_highbits, FxHashMap, (0..).map(usize::swap_bytes));
        $bench_macro!(
            $bench_std_highbits,
            StdHashMap,
            (0..).map(usize::swap_bytes)
        );
        $bench_macro!($bench_fx_random, FxHashMap, RandomKeys::new());
        $bench_macro!($bench_std_random, StdHashMap, RandomKeys::new());
    };
}

macro_rules! bench_insert {
    ($name:ident, $maptype:ident, $keydist:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            b.iter(|| {
                let mut m = $maptype::default();
                for i in ($keydist).take(SIZE) {
                    m.insert(i, i);
                }
                black_box(m);
            })
        }
    };
}

bench_suite!(
    bench_insert,
    insert_fx_serial,
    insert_std_serial,
    insert_fx_highbits,
    insert_std_highbits,
    insert_fx_random,
    insert_std_random
);

macro_rules! bench_insert_erase {
    ($name:ident, $maptype:ident, $keydist:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut m = $maptype::default();
            let mut add_iter = $keydist;
            for i in (&mut add_iter).take(SIZE) {
                m.insert(i, i);
            }
            let mut remove_iter = $keydist;
            b.iter(|| {
                // While keeping the size constant,
                // replace the first keydist with the second.
                for (add, remove) in (&mut add_iter).zip(&mut remove_iter).take(SIZE) {
                    m.insert(add, add);
                    black_box(m.remove(&remove));
                }
            })
        }
    };
}

bench_suite!(
    bench_insert_erase,
    insert_erase_fx_serial,
    insert_erase_std_serial,
    insert_erase_fx_highbits,
    insert_erase_std_highbits,
    insert_erase_fx_random,
    insert_erase_std_random
);

macro_rules! bench_lookup {
    ($name:ident, $maptype:ident, $keydist:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut m = $maptype::default();
            for i in $keydist.take(SIZE) {
                m.insert(i, i);
            }

            b.iter(|| {
                for i in $keydist.take(SIZE) {
                    black_box(m.get(&i));
                }
            })
        }
    };
}

bench_suite!(
    bench_lookup,
    lookup_fx_serial,
    lookup_std_serial,
    lookup_fx_highbits,
    lookup_std_highbits,
    lookup_fx_random,
    lookup_std_random
);

macro_rules! bench_lookup_fail {
    ($name:ident, $maptype:ident, $keydist:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut m = $maptype::default();
            let mut iter = $keydist;
            for i in (&mut iter).take(SIZE) {
                m.insert(i, i);
            }

            b.iter(|| {
                for i in (&mut iter).take(SIZE) {
                    black_box(m.get(&i));
                }
            })
        }
    };
}

bench_suite!(
    bench_lookup_fail,
    lookup_fail_fx_serial,
    lookup_fail_std_serial,
    lookup_fail_fx_highbits,
    lookup_fail_std_highbits,
    lookup_fail_fx_random,
    lookup_fail_std_random
);

macro_rules! bench_iter {
    ($name:ident, $maptype:ident, $keydist:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut m = $maptype::default();
            for i in ($keydist).take(SIZE) {
                m.insert(i, i);
            }

            b.iter(|| {
                for i in &m {
                    black_box(i);
                }
            })
        }
    };
}

bench_suite!(
    bench_iter,
    iter_fx_serial,
    iter_std_serial,
    iter_fx_highbits,
    iter_std_highbits,
    iter_fx_random,
    iter_std_random
);
