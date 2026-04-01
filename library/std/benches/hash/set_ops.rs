use std::collections::HashSet;

use test::Bencher;

#[bench]
fn set_difference(b: &mut Bencher) {
    let small: HashSet<_> = (0..10).collect();
    let large: HashSet<_> = (0..100).collect();

    b.iter(|| small.difference(&large).count());
}

#[bench]
fn set_is_subset(b: &mut Bencher) {
    let small: HashSet<_> = (0..10).collect();
    let large: HashSet<_> = (0..100).collect();

    b.iter(|| small.is_subset(&large));
}

#[bench]
fn set_intersection(b: &mut Bencher) {
    let small: HashSet<_> = (0..10).collect();
    let large: HashSet<_> = (0..100).collect();

    b.iter(|| small.intersection(&large).count());
}

#[bench]
fn set_symmetric_difference(b: &mut Bencher) {
    let small: HashSet<_> = (0..10).collect();
    let large: HashSet<_> = (0..100).collect();

    b.iter(|| small.symmetric_difference(&large).count());
}

#[bench]
fn set_union(b: &mut Bencher) {
    let small: HashSet<_> = (0..10).collect();
    let large: HashSet<_> = (0..100).collect();

    b.iter(|| small.union(&large).count());
}
