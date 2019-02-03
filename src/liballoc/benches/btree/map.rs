use std::iter::Iterator;
use std::vec::Vec;
use std::collections::BTreeMap;

use rand::{Rng, seq::SliceRandom, thread_rng};
use test::{Bencher, black_box};

macro_rules! map_insert_rand_bench {
    ($name: ident, $n: expr, $map: ident) => (
        #[bench]
        pub fn $name(b: &mut Bencher) {
            let n: usize = $n;
            let mut map = $map::new();
            // setup
            let mut rng = thread_rng();

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
    )
}

macro_rules! map_insert_seq_bench {
    ($name: ident, $n: expr, $map: ident) => (
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
    )
}

macro_rules! map_find_rand_bench {
    ($name: ident, $n: expr, $map: ident) => (
        #[bench]
        pub fn $name(b: &mut Bencher) {
            let mut map = $map::new();
            let n: usize = $n;

            // setup
            let mut rng = thread_rng();
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
    )
}

macro_rules! map_find_seq_bench {
    ($name: ident, $n: expr, $map: ident) => (
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
    )
}

map_insert_rand_bench!{insert_rand_100,    100,    BTreeMap}
map_insert_rand_bench!{insert_rand_10_000, 10_000, BTreeMap}

map_insert_seq_bench!{insert_seq_100,    100,    BTreeMap}
map_insert_seq_bench!{insert_seq_10_000, 10_000, BTreeMap}

map_find_rand_bench!{find_rand_100,    100,    BTreeMap}
map_find_rand_bench!{find_rand_10_000, 10_000, BTreeMap}

map_find_seq_bench!{find_seq_100,    100,    BTreeMap}
map_find_seq_bench!{find_seq_10_000, 10_000, BTreeMap}

fn bench_iter(b: &mut Bencher, size: i32) {
    let mut map = BTreeMap::<i32, i32>::new();
    let mut rng = thread_rng();

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
pub fn iter_20(b: &mut Bencher) {
    bench_iter(b, 20);
}

#[bench]
pub fn iter_1000(b: &mut Bencher) {
    bench_iter(b, 1000);
}

#[bench]
pub fn iter_100000(b: &mut Bencher) {
    bench_iter(b, 100000);
}
