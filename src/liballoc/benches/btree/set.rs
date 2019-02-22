use std::collections::BTreeSet;

use rand::{thread_rng, Rng};
use test::{black_box, Bencher};

fn random(n1: u32, n2: u32) -> [BTreeSet<usize>; 2] {
    let mut rng = thread_rng();
    let mut set1 = BTreeSet::new();
    let mut set2 = BTreeSet::new();
    for _ in 0..n1 {
        let i = rng.gen::<usize>();
        set1.insert(i);
    }
    for _ in 0..n2 {
        let i = rng.gen::<usize>();
        set2.insert(i);
    }
    [set1, set2]
}

fn staggered(n1: u32, n2: u32) -> [BTreeSet<u32>; 2] {
    let mut even = BTreeSet::new();
    let mut odd = BTreeSet::new();
    for i in 0..n1 {
        even.insert(i * 2);
    }
    for i in 0..n2 {
        odd.insert(i * 2 + 1);
    }
    [even, odd]
}

fn neg_vs_pos(n1: u32, n2: u32) -> [BTreeSet<i32>; 2] {
    let mut neg = BTreeSet::new();
    let mut pos = BTreeSet::new();
    for i in -(n1 as i32)..=-1 {
        neg.insert(i);
    }
    for i in 1..=(n2 as i32) {
        pos.insert(i);
    }
    [neg, pos]
}

fn pos_vs_neg(n1: u32, n2: u32) -> [BTreeSet<i32>; 2] {
    let mut neg = BTreeSet::new();
    let mut pos = BTreeSet::new();
    for i in -(n1 as i32)..=-1 {
        neg.insert(i);
    }
    for i in 1..=(n2 as i32) {
        pos.insert(i);
    }
    [pos, neg]
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
}

set_intersection_bench! {intersect_random_100,          random(100, 100)}
set_intersection_bench! {intersect_random_10k,          random(10_000, 10_000)}
set_intersection_bench! {intersect_random_10_vs_10k,    random(10, 10_000)}
set_intersection_bench! {intersect_random_10k_vs_10,    random(10_000, 10)}
set_intersection_bench! {intersect_staggered_100,       staggered(100, 100)}
set_intersection_bench! {intersect_staggered_10k,       staggered(10_000, 10_000)}
set_intersection_bench! {intersect_staggered_10_vs_10k, staggered(10, 10_000)}
set_intersection_bench! {intersect_staggered_10k_vs_10, staggered(10_000, 10)}
set_intersection_bench! {intersect_neg_vs_pos_100,      neg_vs_pos(100, 100)}
set_intersection_bench! {intersect_neg_vs_pos_10k,      neg_vs_pos(10_000, 10_000)}
set_intersection_bench! {intersect_neg_vs_pos_10_vs_10k,neg_vs_pos(10, 10_000)}
set_intersection_bench! {intersect_neg_vs_pos_10k_vs_10,neg_vs_pos(10_000, 10)}
set_intersection_bench! {intersect_pos_vs_neg_100,      pos_vs_neg(100, 100)}
set_intersection_bench! {intersect_pos_vs_neg_10k,      pos_vs_neg(10_000, 10_000)}
set_intersection_bench! {intersect_pos_vs_neg_10_vs_10k,pos_vs_neg(10, 10_000)}
set_intersection_bench! {intersect_pos_vs_neg_10k_vs_10,pos_vs_neg(10_000, 10)}
