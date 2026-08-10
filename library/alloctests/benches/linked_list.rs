use std::collections::LinkedList;

use test::{Bencher, black_box};

#[bench]
fn bench_collect_into(b: &mut Bencher) {
    let v = &[0; 64];
    b.iter(|| {
        let _: LinkedList<_> = v.iter().cloned().collect();
    })
}

#[bench]
fn bench_push_front(b: &mut Bencher) {
    let mut m: LinkedList<_> = LinkedList::new();
    b.iter(|| {
        m.push_front(0);
    })
}

#[bench]
fn bench_push_back(b: &mut Bencher) {
    let mut m: LinkedList<_> = LinkedList::new();
    b.iter(|| {
        m.push_back(0);
    })
}

#[bench]
fn bench_push_back_pop_back(b: &mut Bencher) {
    let mut m: LinkedList<_> = LinkedList::new();
    b.iter(|| {
        m.push_back(0);
        m.pop_back();
    })
}

#[bench]
fn bench_push_front_pop_front(b: &mut Bencher) {
    let mut m: LinkedList<_> = LinkedList::new();
    b.iter(|| {
        m.push_front(0);
        m.pop_front();
    })
}

#[bench]
fn bench_iter_count(b: &mut Bencher) {
    let m: LinkedList<_> = (0..128).collect();
    b.iter(|| {
        assert!(black_box(&m).iter().count() == 128);
    })
}

#[bench]
fn bench_iter(b: &mut Bencher) {
    let m: LinkedList<usize> = (0..128).collect();
    b.iter(|| {
        assert!((0..128).sum::<usize>() == black_box(&m).iter().sum());
    })
}

#[bench]
fn bench_iter_mut(b: &mut Bencher) {
    let mut m: LinkedList<usize> = (0..128).collect();
    b.iter(|| {
        black_box(&mut m).iter_mut().for_each(|x| *x += 1);
    })
}
#[bench]
fn bench_iter_rev(b: &mut Bencher) {
    let m: LinkedList<usize> = (0..128).collect();
    b.iter(|| {
        assert!((0..128).sum::<usize>() == black_box(&m).iter().rev().sum());
    })
}
#[bench]
fn bench_iter_mut_rev(b: &mut Bencher) {
    let mut m: LinkedList<usize> = (0..128).collect();
    b.iter(|| {
        black_box(&mut m).iter_mut().rev().for_each(|x| *x += 1);
    })
}
