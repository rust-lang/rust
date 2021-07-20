//! These tests mainly make sure the elements are correctly dropped.

use std::sync::atomic::{AtomicBool, AtomicI64, Ordering::SeqCst};

#[derive(Debug)]
struct DropInfo {
    dropped_twice: AtomicBool,
    alive_count: AtomicI64,
}

impl DropInfo {
    const fn new() -> Self {
        Self { dropped_twice: AtomicBool::new(false), alive_count: AtomicI64::new(0) }
    }

    #[track_caller]
    fn check(&self) {
        assert!(!self.dropped_twice.load(SeqCst), "a value was dropped twice");
        assert_eq!(self.alive_count.load(SeqCst), 0);
    }
}

#[derive(Debug)]
struct DropCheck<'a> {
    index: usize,
    info: &'a DropInfo,
    was_dropped: bool,
}

impl<'a> DropCheck<'a> {
    fn new(index: usize, info: &'a DropInfo) -> Self {
        info.alive_count.fetch_add(1, SeqCst);

        Self { index, info, was_dropped: false }
    }
}

impl Drop for DropCheck<'_> {
    fn drop(&mut self) {
        if self.was_dropped {
            self.info.dropped_twice.store(true, SeqCst);
        }
        self.was_dropped = true;

        self.info.alive_count.fetch_sub(1, SeqCst);
    }
}

fn iter(info: &DropInfo, len: usize, panic_at: usize) -> impl Iterator<Item = DropCheck<'_>> {
    (0..len).map(move |i| {
        if i == panic_at {
            panic!("intended panic");
        }
        DropCheck::new(i, info)
    })
}

#[track_caller]
fn check<const N: usize>(len: usize, panic_at: usize) {
    check_drops(|info| {
        iter(info, len, panic_at).map_windows(|_: &[_; N]| {}).last();
    });
}

#[track_caller]
fn check_drops(f: impl FnOnce(&DropInfo)) {
    let info = DropInfo::new();
    let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        f(&info);
    }));
    info.check();
}

#[test]
fn drop_check_no_iter_panic_n1() {
    check::<1>(0, 100);
    check::<1>(1, 100);
    check::<1>(2, 100);
    check::<1>(13, 100);
}

#[test]
fn drop_check_no_iter_panic_n2() {
    check::<2>(0, 100);
    check::<2>(1, 100);
    check::<2>(2, 100);
    check::<2>(3, 100);
    check::<2>(13, 100);
}

#[test]
fn drop_check_no_iter_panic_n5() {
    check::<5>(0, 100);
    check::<5>(1, 100);
    check::<5>(2, 100);
    check::<5>(13, 100);
    check::<5>(30, 100);
}

#[test]
fn drop_check_panic_in_first_batch() {
    check::<1>(7, 0);

    check::<2>(7, 0);
    check::<2>(7, 1);

    check::<3>(7, 0);
    check::<3>(7, 1);
    check::<3>(7, 2);
}

#[test]
fn drop_check_panic_in_middle() {
    check::<1>(7, 1);
    check::<1>(7, 5);
    check::<1>(7, 6);

    check::<2>(7, 2);
    check::<2>(7, 5);
    check::<2>(7, 6);

    check::<5>(13, 5);
    check::<5>(13, 8);
    check::<5>(13, 12);
}

#[test]
fn drop_check_len_equals_n() {
    check::<1>(1, 100);
    check::<1>(1, 0);

    check::<2>(2, 100);
    check::<2>(2, 0);
    check::<2>(2, 1);

    check::<5>(5, 100);
    check::<5>(5, 0);
    check::<5>(5, 1);
    check::<5>(5, 4);
}

#[test]
fn output_n1() {
    assert_eq!("".chars().map_windows(|[c]| *c).collect::<Vec<_>>(), vec![]);
    assert_eq!("x".chars().map_windows(|[c]| *c).collect::<Vec<_>>(), vec!['x']);
    assert_eq!("abcd".chars().map_windows(|[c]| *c).collect::<Vec<_>>(), vec!['a', 'b', 'c', 'd']);
}

#[test]
fn output_n2() {
    assert_eq!(
        "".chars().map_windows(|a: &[_; 2]| *a).collect::<Vec<_>>(),
        <Vec<[char; 2]>>::new(),
    );
    assert_eq!("ab".chars().map_windows(|a: &[_; 2]| *a).collect::<Vec<_>>(), vec![['a', 'b']]);
    assert_eq!(
        "abcd".chars().map_windows(|a: &[_; 2]| *a).collect::<Vec<_>>(),
        vec![['a', 'b'], ['b', 'c'], ['c', 'd']],
    );
}
