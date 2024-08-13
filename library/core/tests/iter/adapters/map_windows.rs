use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;

#[cfg(not(panic = "abort"))]
mod drop_checks {
    //! These tests mainly make sure the elements are correctly dropped.

    use std::sync::atomic::Ordering::SeqCst;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    #[derive(Debug)]
    struct DropInfo {
        dropped_twice: AtomicBool,
        alive_count: AtomicUsize,
    }

    impl DropInfo {
        const fn new() -> Self {
            Self { dropped_twice: AtomicBool::new(false), alive_count: AtomicUsize::new(0) }
        }

        #[track_caller]
        fn check(&self) {
            assert!(!self.dropped_twice.load(SeqCst), "a value was dropped twice");
            assert_eq!(self.alive_count.load(SeqCst), 0);
        }
    }

    #[derive(Debug)]
    struct DropCheck<'a> {
        info: &'a DropInfo,
        was_dropped: bool,
    }

    impl<'a> DropCheck<'a> {
        fn new(info: &'a DropInfo) -> Self {
            info.alive_count.fetch_add(1, SeqCst);

            Self { info, was_dropped: false }
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
            DropCheck::new(info)
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
    fn no_iter_panic_n1() {
        check::<1>(0, 100);
        check::<1>(1, 100);
        check::<1>(2, 100);
        check::<1>(13, 100);
    }

    #[test]
    fn no_iter_panic_n2() {
        check::<2>(0, 100);
        check::<2>(1, 100);
        check::<2>(2, 100);
        check::<2>(3, 100);
        check::<2>(13, 100);
    }

    #[test]
    fn no_iter_panic_n5() {
        check::<5>(0, 100);
        check::<5>(1, 100);
        check::<5>(2, 100);
        check::<5>(13, 100);
        check::<5>(30, 100);
    }

    #[test]
    fn panic_in_first_batch() {
        check::<1>(7, 0);

        check::<2>(7, 0);
        check::<2>(7, 1);

        check::<3>(7, 0);
        check::<3>(7, 1);
        check::<3>(7, 2);
    }

    #[test]
    fn panic_in_middle() {
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
    fn len_equals_n() {
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

#[test]
fn test_case_from_pr_82413_comment() {
    for () in std::iter::repeat("0".to_owned()).map_windows(|_: &[_; 3]| {}).take(4) {}
}

#[test]
#[should_panic = "array in `Iterator::map_windows` must contain more than 0 elements"]
fn check_zero_window() {
    let _ = std::iter::repeat(0).map_windows(|_: &[_; 0]| ());
}

#[test]
fn test_zero_sized_type() {
    #[derive(Copy, Clone, Debug, Eq, PartialEq)]
    struct Data;
    let data: Vec<_> =
        std::iter::repeat(Data).take(10).map_windows(|arr: &[Data; 5]| *arr).collect();
    assert_eq!(data, [[Data; 5]; 6]);
}

#[test]
#[should_panic = "array size of `Iterator::map_windows` is too large"]
fn test_too_large_array_size() {
    let _ = std::iter::repeat(()).map_windows(|arr: &[(); usize::MAX]| *arr);
}

#[test]
fn test_laziness() {
    let counter = AtomicUsize::new(0);
    let mut iter = (0..5)
        .inspect(|_| {
            counter.fetch_add(1, SeqCst);
        })
        .map_windows(|arr: &[i32; 2]| *arr);
    assert_eq!(counter.load(SeqCst), 0);

    assert_eq!(iter.next(), Some([0, 1]));
    // The first iteration consumes N items (N = 2).
    assert_eq!(counter.load(SeqCst), 2);

    assert_eq!(iter.next(), Some([1, 2]));
    assert_eq!(counter.load(SeqCst), 3);

    assert_eq!(iter.next(), Some([2, 3]));
    assert_eq!(counter.load(SeqCst), 4);

    assert_eq!(iter.next(), Some([3, 4]));
    assert_eq!(counter.load(SeqCst), 5);

    assert_eq!(iter.next(), None);
    assert_eq!(counter.load(SeqCst), 5);
}

#[test]
fn test_size_hint() {
    struct SizeHintCheckHelper((usize, Option<usize>));

    impl Iterator for SizeHintCheckHelper {
        type Item = i32;

        fn next(&mut self) -> Option<i32> {
            let (ref mut lo, ref mut hi) = self.0;
            let next = (*hi != Some(0)).then_some(0);
            *lo = lo.saturating_sub(1);
            if let Some(hi) = hi {
                *hi = hi.saturating_sub(1);
            }
            next
        }

        fn size_hint(&self) -> (usize, Option<usize>) {
            self.0
        }
    }

    fn check_size_hint<const N: usize>(
        size_hint: (usize, Option<usize>),
        mut mapped_size_hint: (usize, Option<usize>),
    ) {
        let mut iter = SizeHintCheckHelper(size_hint);
        let mut mapped_iter = iter.by_ref().map_windows(|_: &[_; N]| ());
        while mapped_iter.size_hint().0 > 0 {
            assert_eq!(mapped_iter.size_hint(), mapped_size_hint);
            assert!(mapped_iter.next().is_some());
            mapped_size_hint.0 -= 1;
            mapped_size_hint.1 = mapped_size_hint.1.map(|hi| hi.saturating_sub(1));
        }
    }

    check_size_hint::<1>((0, None), (0, None));
    check_size_hint::<1>((0, Some(0)), (0, Some(0)));
    check_size_hint::<1>((0, Some(2)), (0, Some(2)));
    check_size_hint::<1>((1, None), (1, None));
    check_size_hint::<1>((1, Some(1)), (1, Some(1)));
    check_size_hint::<1>((1, Some(4)), (1, Some(4)));
    check_size_hint::<1>((5, None), (5, None));
    check_size_hint::<1>((5, Some(5)), (5, Some(5)));
    check_size_hint::<1>((5, Some(10)), (5, Some(10)));

    check_size_hint::<2>((0, None), (0, None));
    check_size_hint::<2>((0, Some(0)), (0, Some(0)));
    check_size_hint::<2>((0, Some(2)), (0, Some(1)));
    check_size_hint::<2>((1, None), (0, None));
    check_size_hint::<2>((1, Some(1)), (0, Some(0)));
    check_size_hint::<2>((1, Some(4)), (0, Some(3)));
    check_size_hint::<2>((5, None), (4, None));
    check_size_hint::<2>((5, Some(5)), (4, Some(4)));
    check_size_hint::<2>((5, Some(10)), (4, Some(9)));

    check_size_hint::<5>((0, None), (0, None));
    check_size_hint::<5>((0, Some(0)), (0, Some(0)));
    check_size_hint::<5>((0, Some(2)), (0, Some(0)));
    check_size_hint::<5>((1, None), (0, None));
    check_size_hint::<5>((1, Some(1)), (0, Some(0)));
    check_size_hint::<5>((1, Some(4)), (0, Some(0)));
    check_size_hint::<5>((5, None), (1, None));
    check_size_hint::<5>((5, Some(5)), (1, Some(1)));
    check_size_hint::<5>((5, Some(10)), (1, Some(6)));
}
