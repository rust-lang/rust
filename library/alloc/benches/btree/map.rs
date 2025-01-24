use std::collections::BTreeMap;
use std::ops::RangeBounds;

use rand::Rng;
use rand::seq::SliceRandom;
use test::{Bencher, black_box};

macro_rules! map_insert_rand_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut Bencher) {
            let n: usize = $n;
            let mut map = $map::new();
            // setup
            let mut rng = crate::bench_rng();

            for _ in 0..n {
                let i = rng.gen::<usize>() % n;
                map.insert(i, i);
            }

            // measure
            b.iter(|| {
                let k = rng.gen::<usize>() % n;
                map.insert(k, k);
                map.remove(&k);
            });
            black_box(map);
        }
    };
}

macro_rules! map_insert_seq_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut Bencher) {
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
    };
}

macro_rules! map_from_iter_rand_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut Bencher) {
            let n: usize = $n;
            // setup
            let mut rng = crate::bench_rng();
            let mut vec = Vec::with_capacity(n);

            for _ in 0..n {
                let i = rng.gen::<usize>() % n;
                vec.push((i, i));
            }

            // measure
            b.iter(|| {
                let map: $map<_, _> = vec.iter().copied().collect();
                black_box(map);
            });
        }
    };
}

macro_rules! map_from_iter_seq_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut Bencher) {
            let n: usize = $n;
            // setup
            let mut vec = Vec::with_capacity(n);

            for i in 0..n {
                vec.push((i, i));
            }

            // measure
            b.iter(|| {
                let map: $map<_, _> = vec.iter().copied().collect();
                black_box(map);
            });
        }
    };
}

macro_rules! map_find_rand_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut Bencher) {
            let mut map = $map::new();
            let n: usize = $n;

            // setup
            let mut rng = crate::bench_rng();
            let mut keys: Vec<_> = (0..n).map(|_| rng.gen::<usize>() % n).collect();

            for &k in &keys {
                map.insert(k, k);
            }

            keys.shuffle(&mut rng);

            // measure
            let mut i = 0;
            b.iter(|| {
                let t = map.get(&keys[i]);
                i = (i + 1) % n;
                black_box(t);
            })
        }
    };
}

macro_rules! map_find_seq_bench {
    ($name: ident, $n: expr, $map: ident) => {
        #[bench]
        pub fn $name(b: &mut Bencher) {
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
    };
}

map_insert_rand_bench! {insert_rand_100,    100,    BTreeMap}
map_insert_rand_bench! {insert_rand_10_000, 10_000, BTreeMap}

map_insert_seq_bench! {insert_seq_100,    100,    BTreeMap}
map_insert_seq_bench! {insert_seq_10_000, 10_000, BTreeMap}

map_from_iter_rand_bench! {from_iter_rand_100,    100,    BTreeMap}
map_from_iter_rand_bench! {from_iter_rand_10_000, 10_000, BTreeMap}

map_from_iter_seq_bench! {from_iter_seq_100,    100,    BTreeMap}
map_from_iter_seq_bench! {from_iter_seq_10_000, 10_000, BTreeMap}

map_find_rand_bench! {find_rand_100,    100,    BTreeMap}
map_find_rand_bench! {find_rand_10_000, 10_000, BTreeMap}

map_find_seq_bench! {find_seq_100,    100,    BTreeMap}
map_find_seq_bench! {find_seq_10_000, 10_000, BTreeMap}

fn bench_iteration(b: &mut Bencher, size: i32) {
    let mut map = BTreeMap::<i32, i32>::new();
    let mut rng = crate::bench_rng();

    for _ in 0..size {
        map.insert(rng.gen(), rng.gen());
    }

    b.iter(|| {
        for entry in &map {
            black_box(entry);
        }
    });
}

#[bench]
pub fn iteration_20(b: &mut Bencher) {
    bench_iteration(b, 20);
}

#[bench]
pub fn iteration_1000(b: &mut Bencher) {
    bench_iteration(b, 1000);
}

#[bench]
pub fn iteration_100000(b: &mut Bencher) {
    bench_iteration(b, 100000);
}

fn bench_iteration_mut(b: &mut Bencher, size: i32) {
    let mut map = BTreeMap::<i32, i32>::new();
    let mut rng = crate::bench_rng();

    for _ in 0..size {
        map.insert(rng.gen(), rng.gen());
    }

    b.iter(|| {
        for kv in map.iter_mut() {
            black_box(kv);
        }
    });
}

#[bench]
pub fn iteration_mut_20(b: &mut Bencher) {
    bench_iteration_mut(b, 20);
}

#[bench]
pub fn iteration_mut_1000(b: &mut Bencher) {
    bench_iteration_mut(b, 1000);
}

#[bench]
pub fn iteration_mut_100000(b: &mut Bencher) {
    bench_iteration_mut(b, 100000);
}

fn bench_first_and_last_nightly(b: &mut Bencher, size: i32) {
    let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();
    b.iter(|| {
        for _ in 0..10 {
            black_box(map.first_key_value());
            black_box(map.last_key_value());
        }
    });
}

fn bench_first_and_last_stable(b: &mut Bencher, size: i32) {
    let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();
    b.iter(|| {
        for _ in 0..10 {
            black_box(map.iter().next());
            black_box(map.iter().next_back());
        }
    });
}

#[bench]
pub fn first_and_last_0_nightly(b: &mut Bencher) {
    bench_first_and_last_nightly(b, 0);
}

#[bench]
pub fn first_and_last_0_stable(b: &mut Bencher) {
    bench_first_and_last_stable(b, 0);
}

#[bench]
pub fn first_and_last_100_nightly(b: &mut Bencher) {
    bench_first_and_last_nightly(b, 100);
}

#[bench]
pub fn first_and_last_100_stable(b: &mut Bencher) {
    bench_first_and_last_stable(b, 100);
}

#[bench]
pub fn first_and_last_10k_nightly(b: &mut Bencher) {
    bench_first_and_last_nightly(b, 10_000);
}

#[bench]
pub fn first_and_last_10k_stable(b: &mut Bencher) {
    bench_first_and_last_stable(b, 10_000);
}

const BENCH_RANGE_SIZE: i32 = 145;
const BENCH_RANGE_COUNT: i32 = BENCH_RANGE_SIZE * (BENCH_RANGE_SIZE - 1) / 2;

fn bench_range<F, R>(b: &mut Bencher, f: F)
where
    F: Fn(i32, i32) -> R,
    R: RangeBounds<i32>,
{
    let map: BTreeMap<_, _> = (0..BENCH_RANGE_SIZE).map(|i| (i, i)).collect();
    b.iter(|| {
        let mut c = 0;
        for i in 0..BENCH_RANGE_SIZE {
            for j in i + 1..BENCH_RANGE_SIZE {
                let _ = black_box(map.range(f(i, j)));
                c += 1;
            }
        }
        debug_assert_eq!(c, BENCH_RANGE_COUNT);
    });
}

#[bench]
pub fn range_included_excluded(b: &mut Bencher) {
    bench_range(b, |i, j| i..j);
}

#[bench]
pub fn range_included_included(b: &mut Bencher) {
    bench_range(b, |i, j| i..=j);
}

#[bench]
pub fn range_included_unbounded(b: &mut Bencher) {
    bench_range(b, |i, _| i..);
}

#[bench]
pub fn range_unbounded_unbounded(b: &mut Bencher) {
    bench_range(b, |_, _| ..);
}

fn bench_iter(b: &mut Bencher, repeats: i32, size: i32) {
    let map: BTreeMap<_, _> = (0..size).map(|i| (i, i)).collect();
    b.iter(|| {
        for _ in 0..repeats {
            let _ = black_box(map.iter());
        }
    });
}

/// Contrast range_unbounded_unbounded with `iter()`.
#[bench]
pub fn range_unbounded_vs_iter(b: &mut Bencher) {
    bench_iter(b, BENCH_RANGE_COUNT, BENCH_RANGE_SIZE);
}

#[bench]
pub fn iter_0(b: &mut Bencher) {
    bench_iter(b, 1_000, 0);
}

#[bench]
pub fn iter_1(b: &mut Bencher) {
    bench_iter(b, 1_000, 1);
}

#[bench]
pub fn iter_100(b: &mut Bencher) {
    bench_iter(b, 1_000, 100);
}

#[bench]
pub fn iter_10k(b: &mut Bencher) {
    bench_iter(b, 1_000, 10_000);
}

#[bench]
pub fn iter_1m(b: &mut Bencher) {
    bench_iter(b, 1_000, 1_000_000);
}

const FAT: usize = 256;

// The returned map has small keys and values.
// Benchmarks on it have a counterpart in set.rs with the same keys and no values at all.
fn slim_map(n: usize) -> BTreeMap<usize, usize> {
    (0..n).map(|i| (i, i)).collect::<BTreeMap<_, _>>()
}

// The returned map has small keys and large values.
fn fat_val_map(n: usize) -> BTreeMap<usize, [usize; FAT]> {
    (0..n).map(|i| (i, [i; FAT])).collect::<BTreeMap<_, _>>()
}

#[bench]
pub fn clone_slim_100(b: &mut Bencher) {
    let src = slim_map(100);
    b.iter(|| src.clone())
}

#[bench]
pub fn clone_slim_100_and_clear(b: &mut Bencher) {
    let src = slim_map(100);
    b.iter(|| src.clone().clear())
}

#[bench]
pub fn clone_slim_100_and_drain_all(b: &mut Bencher) {
    let src = slim_map(100);
    b.iter(|| src.clone().extract_if(|_, _| true).count())
}

#[bench]
pub fn clone_slim_100_and_drain_half(b: &mut Bencher) {
    let src = slim_map(100);
    b.iter(|| {
        let mut map = src.clone();
        assert_eq!(map.extract_if(|i, _| i % 2 == 0).count(), 100 / 2);
        assert_eq!(map.len(), 100 / 2);
    })
}

#[bench]
pub fn clone_slim_100_and_into_iter(b: &mut Bencher) {
    let src = slim_map(100);
    b.iter(|| src.clone().into_iter().count())
}

#[bench]
pub fn clone_slim_100_and_pop_all(b: &mut Bencher) {
    let src = slim_map(100);
    b.iter(|| {
        let mut map = src.clone();
        while map.pop_first().is_some() {}
        map
    });
}

#[bench]
pub fn clone_slim_100_and_remove_all(b: &mut Bencher) {
    let src = slim_map(100);
    b.iter(|| {
        let mut map = src.clone();
        while let Some(elt) = map.iter().map(|(&i, _)| i).next() {
            let v = map.remove(&elt);
            debug_assert!(v.is_some());
        }
        map
    });
}

#[bench]
pub fn clone_slim_100_and_remove_half(b: &mut Bencher) {
    let src = slim_map(100);
    b.iter(|| {
        let mut map = src.clone();
        for i in (0..100).step_by(2) {
            let v = map.remove(&i);
            debug_assert!(v.is_some());
        }
        assert_eq!(map.len(), 100 / 2);
        map
    })
}

#[bench]
pub fn clone_slim_10k(b: &mut Bencher) {
    let src = slim_map(10_000);
    b.iter(|| src.clone())
}

#[bench]
pub fn clone_slim_10k_and_clear(b: &mut Bencher) {
    let src = slim_map(10_000);
    b.iter(|| src.clone().clear())
}

#[bench]
pub fn clone_slim_10k_and_drain_all(b: &mut Bencher) {
    let src = slim_map(10_000);
    b.iter(|| src.clone().extract_if(|_, _| true).count())
}

#[bench]
pub fn clone_slim_10k_and_drain_half(b: &mut Bencher) {
    let src = slim_map(10_000);
    b.iter(|| {
        let mut map = src.clone();
        assert_eq!(map.extract_if(|i, _| i % 2 == 0).count(), 10_000 / 2);
        assert_eq!(map.len(), 10_000 / 2);
    })
}

#[bench]
pub fn clone_slim_10k_and_into_iter(b: &mut Bencher) {
    let src = slim_map(10_000);
    b.iter(|| src.clone().into_iter().count())
}

#[bench]
pub fn clone_slim_10k_and_pop_all(b: &mut Bencher) {
    let src = slim_map(10_000);
    b.iter(|| {
        let mut map = src.clone();
        while map.pop_first().is_some() {}
        map
    });
}

#[bench]
pub fn clone_slim_10k_and_remove_all(b: &mut Bencher) {
    let src = slim_map(10_000);
    b.iter(|| {
        let mut map = src.clone();
        while let Some(elt) = map.iter().map(|(&i, _)| i).next() {
            let v = map.remove(&elt);
            debug_assert!(v.is_some());
        }
        map
    });
}

#[bench]
pub fn clone_slim_10k_and_remove_half(b: &mut Bencher) {
    let src = slim_map(10_000);
    b.iter(|| {
        let mut map = src.clone();
        for i in (0..10_000).step_by(2) {
            let v = map.remove(&i);
            debug_assert!(v.is_some());
        }
        assert_eq!(map.len(), 10_000 / 2);
        map
    })
}

#[bench]
pub fn clone_fat_val_100(b: &mut Bencher) {
    let src = fat_val_map(100);
    b.iter(|| src.clone())
}

#[bench]
pub fn clone_fat_val_100_and_clear(b: &mut Bencher) {
    let src = fat_val_map(100);
    b.iter(|| src.clone().clear())
}

#[bench]
pub fn clone_fat_val_100_and_drain_all(b: &mut Bencher) {
    let src = fat_val_map(100);
    b.iter(|| src.clone().extract_if(|_, _| true).count())
}

#[bench]
pub fn clone_fat_val_100_and_drain_half(b: &mut Bencher) {
    let src = fat_val_map(100);
    b.iter(|| {
        let mut map = src.clone();
        assert_eq!(map.extract_if(|i, _| i % 2 == 0).count(), 100 / 2);
        assert_eq!(map.len(), 100 / 2);
    })
}

#[bench]
pub fn clone_fat_val_100_and_into_iter(b: &mut Bencher) {
    let src = fat_val_map(100);
    b.iter(|| src.clone().into_iter().count())
}

#[bench]
pub fn clone_fat_val_100_and_pop_all(b: &mut Bencher) {
    let src = fat_val_map(100);
    b.iter(|| {
        let mut map = src.clone();
        while map.pop_first().is_some() {}
        map
    });
}

#[bench]
pub fn clone_fat_val_100_and_remove_all(b: &mut Bencher) {
    let src = fat_val_map(100);
    b.iter(|| {
        let mut map = src.clone();
        while let Some(elt) = map.iter().map(|(&i, _)| i).next() {
            let v = map.remove(&elt);
            debug_assert!(v.is_some());
        }
        map
    });
}

#[bench]
pub fn clone_fat_val_100_and_remove_half(b: &mut Bencher) {
    let src = fat_val_map(100);
    b.iter(|| {
        let mut map = src.clone();
        for i in (0..100).step_by(2) {
            let v = map.remove(&i);
            debug_assert!(v.is_some());
        }
        assert_eq!(map.len(), 100 / 2);
        map
    })
}
