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
    remaining: usize,
    state: usize,
}

impl RandomKeys {
    fn new(size: usize) -> Self {
        RandomKeys {
            remaining: size,
            state: 1,
        }
    }

    // Produce a different set of random values.
    fn new2(size: usize) -> Self {
        RandomKeys {
            remaining: size,
            state: 2,
        }
    }
}

impl Iterator for RandomKeys {
    type Item = usize;
    fn next(&mut self) -> Option<usize> {
        if self.remaining == 0 {
            None
        } else {
            self.remaining -= 1;
            // Multiply by some 32 bit prime.
            self.state = self.state.wrapping_mul(3787392781);
            // Mix in to the bottom bits which are constant mod powers of 2.
            Some(self.state ^ (self.state >> 4))
        }
    }
}

macro_rules! bench_insert {
    ($name:ident, $maptype:ident, $keydist:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            b.iter(|| {
                let mut m = $maptype::default();
                for i in $keydist {
                    m.insert(i, i);
                }
                black_box(m);
            })
        }
    };
}

bench_insert!(insert_fx_serial, FxHashMap, 0..SIZE);
bench_insert!(insert_std_serial, StdHashMap, 0..SIZE);
bench_insert!(
    insert_fx_highbits,
    FxHashMap,
    (0..SIZE).map(usize::swap_bytes)
);
bench_insert!(
    insert_std_highbits,
    StdHashMap,
    (0..SIZE).map(usize::swap_bytes)
);
bench_insert!(insert_fx_random, FxHashMap, RandomKeys::new(SIZE));
bench_insert!(insert_std_random, StdHashMap, RandomKeys::new(SIZE));

macro_rules! bench_insert_erase {
    ($name:ident, $maptype:ident, $keydist:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            b.iter(|| {
                let mut m = $maptype::default();
                for i in $keydist {
                    m.insert(i, i);
                }
                black_box(&mut m);
                for i in $keydist {
                    m.remove(&i);
                }
                black_box(m);
            })
        }
    };
}

bench_insert_erase!(insert_erase_fx_serial, FxHashMap, 0..SIZE);
bench_insert_erase!(insert_erase_std_serial, StdHashMap, 0..SIZE);
bench_insert_erase!(
    insert_erase_fx_highbits,
    FxHashMap,
    (0..SIZE).map(usize::swap_bytes)
);
bench_insert_erase!(
    insert_erase_std_highbits,
    StdHashMap,
    (0..SIZE).map(usize::swap_bytes)
);
bench_insert_erase!(insert_erase_fx_random, FxHashMap, RandomKeys::new(SIZE));
bench_insert_erase!(insert_erase_std_random, StdHashMap, RandomKeys::new(SIZE));

macro_rules! bench_lookup {
    ($name:ident, $maptype:ident, $keydist:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut m = $maptype::default();
            for i in $keydist {
                m.insert(i, i);
            }

            b.iter(|| {
                for i in $keydist {
                    black_box(m.get(&i));
                }
            })
        }
    };
}

bench_lookup!(lookup_fx_serial, FxHashMap, 0..SIZE);
bench_lookup!(lookup_std_serial, StdHashMap, 0..SIZE);
bench_lookup!(
    lookup_fx_highbits,
    FxHashMap,
    (0..SIZE).map(usize::swap_bytes)
);
bench_lookup!(
    lookup_std_highbits,
    StdHashMap,
    (0..SIZE).map(usize::swap_bytes)
);
bench_lookup!(lookup_fx_random, FxHashMap, RandomKeys::new(SIZE));
bench_lookup!(lookup_std_random, StdHashMap, RandomKeys::new(SIZE));

macro_rules! bench_lookup_fail {
    ($name:ident, $maptype:ident, $keydist:expr, $keydist2:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut m = $maptype::default();
            for i in $keydist {
                m.insert(i, i);
            }

            b.iter(|| {
                for i in $keydist2 {
                    black_box(m.get(&i));
                }
            })
        }
    };
}

bench_lookup_fail!(lookup_fail_fx_serial, FxHashMap, 0..SIZE, SIZE..SIZE * 2);
bench_lookup_fail!(lookup_fail_std_serial, StdHashMap, 0..SIZE, SIZE..SIZE * 2);
bench_lookup_fail!(
    lookup_fail_fx_highbits,
    FxHashMap,
    (0..SIZE).map(usize::swap_bytes),
    (SIZE..SIZE * 2).map(usize::swap_bytes)
);
bench_lookup_fail!(
    lookup_fail_std_highbits,
    StdHashMap,
    (0..SIZE).map(usize::swap_bytes),
    (SIZE..SIZE * 2).map(usize::swap_bytes)
);
bench_lookup_fail!(
    lookup_fail_fx_random,
    FxHashMap,
    RandomKeys::new(SIZE),
    RandomKeys::new2(SIZE)
);
bench_lookup_fail!(
    lookup_fail_std_random,
    StdHashMap,
    RandomKeys::new(SIZE),
    RandomKeys::new2(SIZE)
);

macro_rules! bench_iter {
    ($name:ident, $maptype:ident, $keydist:expr) => {
        #[bench]
        fn $name(b: &mut Bencher) {
            let mut m = $maptype::default();
            for i in $keydist {
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

bench_iter!(iter_fx_serial, FxHashMap, 0..SIZE);
bench_iter!(iter_std_serial, StdHashMap, 0..SIZE);
bench_iter!(
    iter_fx_highbits,
    FxHashMap,
    (0..SIZE).map(usize::swap_bytes)
);
bench_iter!(
    iter_std_highbits,
    StdHashMap,
    (0..SIZE).map(usize::swap_bytes)
);
bench_iter!(iter_fx_random, FxHashMap, RandomKeys::new(SIZE));
bench_iter!(iter_std_random, StdHashMap, RandomKeys::new(SIZE));
