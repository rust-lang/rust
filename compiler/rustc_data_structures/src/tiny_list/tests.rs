use super::*;

extern crate test;
use test::{black_box, Bencher};

impl<T> TinyList<T> {
    fn len(&self) -> usize {
        let (mut elem, mut count) = (self.head.as_ref(), 0);
        while let Some(e) = elem {
            count += 1;
            elem = e.next.as_deref();
        }
        count
    }
}

#[test]
fn test_contains_and_insert() {
    fn do_insert(i: u32) -> bool {
        i % 2 == 0
    }

    let mut list = TinyList::new();

    for i in 0..10 {
        for j in 0..i {
            if do_insert(j) {
                assert!(list.contains(&j));
            } else {
                assert!(!list.contains(&j));
            }
        }

        assert!(!list.contains(&i));

        if do_insert(i) {
            list.insert(i);
            assert!(list.contains(&i));
        }
    }
}

#[test]
fn test_remove_first() {
    let mut list = TinyList::new();
    list.insert(1);
    list.insert(2);
    list.insert(3);
    list.insert(4);
    assert_eq!(list.len(), 4);

    assert!(list.remove(&4));
    assert!(!list.contains(&4));

    assert_eq!(list.len(), 3);
    assert!(list.contains(&1));
    assert!(list.contains(&2));
    assert!(list.contains(&3));
}

#[test]
fn test_remove_last() {
    let mut list = TinyList::new();
    list.insert(1);
    list.insert(2);
    list.insert(3);
    list.insert(4);
    assert_eq!(list.len(), 4);

    assert!(list.remove(&1));
    assert!(!list.contains(&1));

    assert_eq!(list.len(), 3);
    assert!(list.contains(&2));
    assert!(list.contains(&3));
    assert!(list.contains(&4));
}

#[test]
fn test_remove_middle() {
    let mut list = TinyList::new();
    list.insert(1);
    list.insert(2);
    list.insert(3);
    list.insert(4);
    assert_eq!(list.len(), 4);

    assert!(list.remove(&2));
    assert!(!list.contains(&2));

    assert_eq!(list.len(), 3);
    assert!(list.contains(&1));
    assert!(list.contains(&3));
    assert!(list.contains(&4));
}

#[test]
fn test_remove_single() {
    let mut list = TinyList::new();
    list.insert(1);
    assert_eq!(list.len(), 1);

    assert!(list.remove(&1));
    assert!(!list.contains(&1));

    assert_eq!(list.len(), 0);
}

#[bench]
fn bench_insert_empty(b: &mut Bencher) {
    b.iter(|| {
        let mut list = black_box(TinyList::new());
        list.insert(1);
        list
    })
}

#[bench]
fn bench_insert_one(b: &mut Bencher) {
    b.iter(|| {
        let mut list = black_box(TinyList::new_single(0));
        list.insert(1);
        list
    })
}

#[bench]
fn bench_contains_empty(b: &mut Bencher) {
    b.iter(|| black_box(TinyList::new()).contains(&1));
}

#[bench]
fn bench_contains_unknown(b: &mut Bencher) {
    b.iter(|| black_box(TinyList::new_single(0)).contains(&1));
}

#[bench]
fn bench_contains_one(b: &mut Bencher) {
    b.iter(|| black_box(TinyList::new_single(1)).contains(&1));
}

#[bench]
fn bench_remove_empty(b: &mut Bencher) {
    b.iter(|| black_box(TinyList::new()).remove(&1));
}

#[bench]
fn bench_remove_unknown(b: &mut Bencher) {
    b.iter(|| black_box(TinyList::new_single(0)).remove(&1));
}

#[bench]
fn bench_remove_one(b: &mut Bencher) {
    b.iter(|| black_box(TinyList::new_single(1)).remove(&1));
}
