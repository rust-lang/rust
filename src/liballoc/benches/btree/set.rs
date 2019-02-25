use std::collections::BTreeSet;

use rand::{thread_rng, Rng};
use test::{black_box, Bencher};

fn random(n1: usize, n2: usize) -> [BTreeSet<usize>; 2] {
    let mut rng = thread_rng();
    let mut sets = [BTreeSet::new(), BTreeSet::new()];
    for i in 0..2 {
        while sets[i].len() < [n1,n2][i] {
            sets[i].insert(rng.gen());
        }
    }
    assert_eq!(sets[0].len(), n1);
    assert_eq!(sets[1].len(), n2);
    sets
}

fn stagger(n1: usize, factor: usize) -> [BTreeSet<u32>; 2] {
    let n2 = n1 * factor;
    let mut sets = [BTreeSet::new(), BTreeSet::new()];
    for i in 0..(n1+n2) {
        let b = i % (factor + 1) != 0;
        sets[b as usize].insert(i as u32);
    }
    assert_eq!(sets[0].len(), n1);
    assert_eq!(sets[1].len(), n2);
    sets
}

fn neg_vs_pos(n1: usize, n2: usize) -> [BTreeSet<i32>; 2] {
    let mut neg = BTreeSet::new();
    let mut pos = BTreeSet::new();
    for i in -(n1 as i32)..=-1 {
        neg.insert(i);
    }
    for i in 1..=(n2 as i32) {
        pos.insert(i);
    }
    assert_eq!(neg.len(), n1);
    assert_eq!(pos.len(), n2);
    [neg, pos]
}

fn pos_vs_neg(n1: usize, n2: usize) -> [BTreeSet<i32>; 2] {
    let mut sets = neg_vs_pos(n2, n1);
    sets.reverse();
    assert_eq!(sets[0].len(), n1);
    assert_eq!(sets[1].len(), n2);
    sets
}

macro_rules! set_intersection_bench {
    ($name: ident, $sets: expr) => {
        #[bench]
        pub fn $name(b: &mut Bencher) {
            // setup
            let sets = $sets;

            // measure
            b.iter(|| {
                let x = sets[0].intersection(&sets[1]).count();
                black_box(x);
            })
        }
    };
    ($name: ident, $sets: expr, $intersection_kind: ident) => {
        #[bench]
        pub fn $name(b: &mut Bencher) {
            // setup
            let sets = $sets;
            assert!(sets[0].len() >= 1);
            assert!(sets[1].len() >= sets[0].len());

            // measure
            b.iter(|| {
                let x = BTreeSet::$intersection_kind(&sets[0], &sets[1]).count();
                black_box(x);
            })
        }
    };
}

set_intersection_bench! {intersect_neg_vs_pos_100,          neg_vs_pos(100, 100)}
set_intersection_bench! {intersect_neg_vs_pos_10k,          neg_vs_pos(10_000, 10_000)}
set_intersection_bench! {intersect_neg_vs_pos_10_vs_10k,    neg_vs_pos(10, 10_000)}
set_intersection_bench! {intersect_neg_vs_pos_10k_vs_10,    neg_vs_pos(10_000, 10)}
set_intersection_bench! {intersect_pos_vs_neg_100,          pos_vs_neg(100, 100)}
set_intersection_bench! {intersect_pos_vs_neg_10k,          pos_vs_neg(10_000, 10_000)}
set_intersection_bench! {intersect_pos_vs_neg_10_vs_10k,    pos_vs_neg(10, 10_000)}
set_intersection_bench! {intersect_pos_vs_neg_10k_vs_10,    pos_vs_neg(10_000, 10)}
set_intersection_bench! {intersect_random_100,              random(100, 100)}
set_intersection_bench! {intersect_random_10k,              random(10_000, 10_000)}
set_intersection_bench! {intersect_random_10_vs_10k,        random(10, 10_000)}
set_intersection_bench! {intersect_random_10k_vs_10,        random(10_000, 10)}
set_intersection_bench! {intersect_stagger_10k,             stagger(10_000, 1)}
set_intersection_bench! {intersect_stagger_100,             stagger(100, 1)}
set_intersection_bench! {intersect_stagger_100_df1,         stagger(100, 1 << 1)}
set_intersection_bench! {intersect_stagger_100_df1_stitch,  stagger(100, 1 << 1), intersection_stitch}
set_intersection_bench! {intersect_stagger_100_df1_search,  stagger(100, 1 << 1), intersection_search}
set_intersection_bench! {intersect_stagger_100_df2,         stagger(100, 1 << 2)}
set_intersection_bench! {intersect_stagger_100_df2_stitch,  stagger(100, 1 << 2), intersection_stitch}
set_intersection_bench! {intersect_stagger_100_df2_search,  stagger(100, 1 << 2), intersection_search}
set_intersection_bench! {intersect_stagger_100_df3,         stagger(100, 1 << 3)}
set_intersection_bench! {intersect_stagger_100_df3_stitch,  stagger(100, 1 << 3), intersection_stitch}
set_intersection_bench! {intersect_stagger_100_df3_search,  stagger(100, 1 << 3), intersection_search}
set_intersection_bench! {intersect_stagger_100_df4,         stagger(100, 1 << 4)}
set_intersection_bench! {intersect_stagger_100_df4_stitch,  stagger(100, 1 << 4), intersection_stitch}
set_intersection_bench! {intersect_stagger_100_df4_search,  stagger(100, 1 << 4), intersection_search}
set_intersection_bench! {intersect_stagger_100_df5,         stagger(100, 1 << 5)}
set_intersection_bench! {intersect_stagger_100_df5_stitch,  stagger(100, 1 << 5), intersection_stitch}
set_intersection_bench! {intersect_stagger_100_df5_search,  stagger(100, 1 << 5), intersection_search}
set_intersection_bench! {intersect_stagger_100_df6,         stagger(100, 1 << 6)}
set_intersection_bench! {intersect_stagger_100_df6_stitch,  stagger(100, 1 << 6), intersection_stitch}
set_intersection_bench! {intersect_stagger_100_df6_search,  stagger(100, 1 << 6), intersection_search}
