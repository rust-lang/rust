use std::collections::BTreeSet;

use rand::{thread_rng, Rng};
use test::{black_box, Bencher};

fn random(n: usize) -> BTreeSet<usize> {
    let mut rng = thread_rng();
    let mut set = BTreeSet::new();
    while set.len() < n {
        set.insert(rng.gen());
    }
    assert_eq!(set.len(), n);
    set
}

fn neg(n: usize) -> BTreeSet<i32> {
    let mut set = BTreeSet::new();
    for i in -(n as i32)..=-1 {
        set.insert(i);
    }
    assert_eq!(set.len(), n);
    set
}

fn pos(n: usize) -> BTreeSet<i32> {
    let mut set = BTreeSet::new();
    for i in 1..=(n as i32) {
        set.insert(i);
    }
    assert_eq!(set.len(), n);
    set
}


fn stagger(n1: usize, factor: usize) -> [BTreeSet<u32>; 2] {
    let n2 = n1 * factor;
    let mut sets = [BTreeSet::new(), BTreeSet::new()];
    for i in 0..(n1 + n2) {
        let b = i % (factor + 1) != 0;
        sets[b as usize].insert(i as u32);
    }
    assert_eq!(sets[0].len(), n1);
    assert_eq!(sets[1].len(), n2);
    sets
}

macro_rules! set_bench {
    ($name: ident, $set_func: ident, $result_func: ident, $sets: expr) => {
        #[bench]
        pub fn $name(b: &mut Bencher) {
            // setup
            let sets = $sets;

            // measure
            b.iter(|| {
                let x = sets[0].$set_func(&sets[1]).$result_func();
                black_box(x);
            })
        }
    };
}

set_bench! {intersection_100_neg_vs_100_pos, intersection, count, [neg(100), pos(100)]}
set_bench! {intersection_100_neg_vs_10k_pos, intersection, count, [neg(100), pos(10_000)]}
set_bench! {intersection_100_pos_vs_100_neg, intersection, count, [pos(100), neg(100)]}
set_bench! {intersection_100_pos_vs_10k_neg, intersection, count, [pos(100), neg(10_000)]}
set_bench! {intersection_10k_neg_vs_100_pos, intersection, count, [neg(10_000), pos(100)]}
set_bench! {intersection_10k_neg_vs_10k_pos, intersection, count, [neg(10_000), pos(10_000)]}
set_bench! {intersection_10k_pos_vs_100_neg, intersection, count, [pos(10_000), neg(100)]}
set_bench! {intersection_10k_pos_vs_10k_neg, intersection, count, [pos(10_000), neg(10_000)]}
set_bench! {intersection_random_100_vs_100, intersection, count, [random(100), random(100)]}
set_bench! {intersection_random_100_vs_10k, intersection, count, [random(100), random(10_000)]}
set_bench! {intersection_random_10k_vs_100, intersection, count, [random(10_000), random(100)]}
set_bench! {intersection_random_10k_vs_10k, intersection, count, [random(10_000), random(10_000)]}
set_bench! {intersection_staggered_100_vs_100, intersection, count, stagger(100, 1)}
set_bench! {intersection_staggered_10k_vs_10k, intersection, count, stagger(10_000, 1)}
set_bench! {intersection_staggered_100_vs_10k, intersection, count, stagger(100, 100)}
set_bench! {difference_random_100_vs_100, difference, count, [random(100), random(100)]}
set_bench! {difference_random_100_vs_10k, difference, count, [random(100), random(10_000)]}
set_bench! {difference_random_10k_vs_100, difference, count, [random(10_000), random(100)]}
set_bench! {difference_random_10k_vs_10k, difference, count, [random(10_000), random(10_000)]}
set_bench! {difference_staggered_100_vs_100, difference, count, stagger(100, 1)}
set_bench! {difference_staggered_10k_vs_10k, difference, count, stagger(10_000, 1)}
set_bench! {difference_staggered_100_vs_10k, difference, count, stagger(100, 100)}
set_bench! {is_subset_100_vs_100, is_subset, clone, [pos(100), pos(100)]}
set_bench! {is_subset_100_vs_10k, is_subset, clone, [pos(100), pos(10_000)]}
set_bench! {is_subset_10k_vs_100, is_subset, clone, [pos(10_000), pos(100)]}
set_bench! {is_subset_10k_vs_10k, is_subset, clone, [pos(10_000), pos(10_000)]}
