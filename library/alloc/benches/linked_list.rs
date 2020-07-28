use std::collections::LinkedList;
use test::Bencher;

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
fn bench_iter(b: &mut Bencher) {
    let v = &[0; 128];
    let m: LinkedList<_> = v.iter().cloned().collect();
    b.iter(|| {
        assert!(m.iter().count() == 128);
    })
}
#[bench]
fn bench_iter_mut(b: &mut Bencher) {
    let v = &[0; 128];
    let mut m: LinkedList<_> = v.iter().cloned().collect();
    b.iter(|| {
        assert!(m.iter_mut().count() == 128);
    })
}
#[bench]
fn bench_iter_rev(b: &mut Bencher) {
    let v = &[0; 128];
    let m: LinkedList<_> = v.iter().cloned().collect();
    b.iter(|| {
        assert!(m.iter().rev().count() == 128);
    })
}
#[bench]
fn bench_iter_mut_rev(b: &mut Bencher) {
    let v = &[0; 128];
    let mut m: LinkedList<_> = v.iter().cloned().collect();
    b.iter(|| {
        assert!(m.iter_mut().rev().count() == 128);
    })
}
