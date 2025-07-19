use core::iter::*;

use super::*;

#[test]
fn test_zip_nth() {
    let xs = [0, 1, 2, 4, 5];
    let ys = [10, 11, 12];

    let mut it = xs.iter().zip(&ys);
    assert_eq!(it.nth(0), Some((&0, &10)));
    assert_eq!(it.nth(1), Some((&2, &12)));
    assert_eq!(it.nth(0), None);

    let mut it = xs.iter().zip(&ys);
    assert_eq!(it.nth(3), None);

    let mut it = ys.iter().zip(&xs);
    assert_eq!(it.nth(3), None);
}

#[test]
fn test_zip_nth_side_effects() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let value = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4, 5, 6, 7, 8].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }))
        .skip(1)
        .nth(3);
    assert_eq!(value, Some((50, 6000)));
    assert_eq!(a, vec![1, 2, 3, 4, 5]);
    assert_eq!(b, vec![200, 300, 400, 500, 600]);
}

#[test]
fn test_zip_next_back_side_effects() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut iter = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4, 5, 6, 7, 8].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }));

    // The second iterator is one item longer, so `next_back` is called on it
    // one more time.
    assert_eq!(iter.next_back(), Some((60, 7000)));
    assert_eq!(iter.next_back(), Some((50, 6000)));
    assert_eq!(iter.next_back(), Some((40, 5000)));
    assert_eq!(iter.next_back(), Some((30, 4000)));
    assert_eq!(a, vec![6, 5, 4, 3]);
    assert_eq!(b, vec![800, 700, 600, 500, 400]);
}

#[test]
fn test_zip_nth_back_side_effects() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let value = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4, 5, 6, 7, 8].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }))
        .nth_back(3);
    assert_eq!(value, Some((30, 4000)));
    assert_eq!(a, vec![6, 5, 4, 3]);
    assert_eq!(b, vec![800, 700, 600, 500, 400]);
}

#[test]
fn test_zip_next_back_side_effects_exhausted() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut iter = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }));

    iter.next();
    iter.next();
    iter.next();
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next_back(), None);

    assert!(a.starts_with(&[1, 2, 3]));
    let a_len = a.len();
    // Tail-side-effects of forward-iteration are "at most one" per next().
    // And for reverse iteration we don't guarantee much either.
    // But we can put some bounds on the possible behaviors.
    assert!(a_len <= 6);
    assert!(a_len >= 3);
    a.sort();
    assert_eq!(a, &[1, 2, 3, 4, 5, 6][..a.len()]);

    assert_eq!(b, vec![200, 300, 400]);
}

#[test]
fn test_zip_cloned_sideffectful() {
    let xs = [CountClone::new(), CountClone::new(), CountClone::new(), CountClone::new()];
    let ys = [CountClone::new(), CountClone::new()];

    for _ in xs.iter().cloned().zip(ys.iter().cloned()) {}

    // Zip documentation permits either case.
    assert!([&[1, 1, 1, 0], &[1, 1, 0, 0]].iter().any(|v| &xs == *v));
    assert_eq!(&ys, &[1, 1][..]);

    let xs = [CountClone::new(), CountClone::new()];
    let ys = [CountClone::new(), CountClone::new(), CountClone::new(), CountClone::new()];

    for _ in xs.iter().cloned().zip(ys.iter().cloned()) {}

    assert_eq!(&xs, &[1, 1][..]);
    assert_eq!(&ys, &[1, 1, 0, 0][..]);
}

#[test]
fn test_zip_map_sideffectful() {
    let mut xs = [0; 6];
    let mut ys = [0; 4];

    for _ in xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1)) {}

    // Zip documentation permits either case.
    assert!([&[1, 1, 1, 1, 1, 0], &[1, 1, 1, 1, 0, 0]].iter().any(|v| &xs == *v));
    assert_eq!(&ys, &[1, 1, 1, 1]);

    let mut xs = [0; 4];
    let mut ys = [0; 6];

    for _ in xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1)) {}

    assert_eq!(&xs, &[1, 1, 1, 1]);
    assert_eq!(&ys, &[1, 1, 1, 1, 0, 0]);
}

#[test]
fn test_zip_map_rev_sideffectful() {
    let mut xs = [0; 6];
    let mut ys = [0; 4];

    {
        let mut it = xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1));
        it.next_back();
    }
    assert_eq!(&xs, &[0, 0, 0, 1, 1, 1]);
    assert_eq!(&ys, &[0, 0, 0, 1]);

    let mut xs = [0; 6];
    let mut ys = [0; 4];

    {
        let mut it = xs.iter_mut().map(|x| *x += 1).zip(ys.iter_mut().map(|y| *y += 1));
        // the current impl only trims the tails if the iterator isn't exhausted
        (&mut it).take(3).count();
        it.next_back();
    }
    assert_eq!(&xs, &[1, 1, 1, 1, 1, 1]);
    assert_eq!(&ys, &[1, 1, 1, 1]);
}

#[test]
fn test_zip_nested_sideffectful() {
    let mut xs = [0; 6];
    let ys = [0; 4];

    {
        // test that it has the side effect nested inside enumerate
        let it = xs.iter_mut().map(|x| *x = 1).enumerate().zip(&ys);
        it.count();
    }
    let length_aware = &xs == &[1, 1, 1, 1, 0, 0];
    let probe_first = &xs == &[1, 1, 1, 1, 1, 0];

    // either implementation is valid according to zip documentation
    assert!(length_aware || probe_first);
}

#[test]
fn test_zip_nth_back_side_effects_exhausted() {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut iter = [1, 2, 3, 4, 5, 6]
        .iter()
        .cloned()
        .map(|n| {
            a.push(n);
            n * 10
        })
        .zip([2, 3, 4].iter().cloned().map(|n| {
            b.push(n * 100);
            n * 1000
        }));

    iter.next();
    iter.next();
    iter.next();
    assert_eq!(iter.next(), None);
    assert_eq!(iter.nth_back(0), None);
    assert!(a.starts_with(&[1, 2, 3]));
    let a_len = a.len();
    // Tail-side-effects of forward-iteration are "at most one" per next().
    // And for reverse iteration we don't guarantee much either.
    // But we can put some bounds on the possible behaviors.
    assert!(a_len <= 6);
    assert!(a_len >= 3);
    a.sort();
    assert_eq!(a, &[1, 2, 3, 4, 5, 6][..a.len()]);

    assert_eq!(b, vec![200, 300, 400]);
}

#[test]
fn test_zip_trusted_random_access_composition() {
    let a = [0, 1, 2, 3, 4];
    let b = a;
    let c = a;

    let a = a.iter().copied();
    let b = b.iter().copied();
    let mut c = c.iter().copied();
    c.next();

    let mut z1 = a.zip(b);
    assert_eq!(z1.next().unwrap(), (0, 0));

    let mut z2 = z1.zip(c);
    fn assert_trusted_random_access<T: TrustedRandomAccess>(_a: &T) {}
    assert_trusted_random_access(&z2);
    assert_eq!(z2.next().unwrap(), ((1, 1), 1));
}

#[test]
fn test_double_ended_zip() {
    let xs = [1, 2, 3, 4, 5, 6];
    let ys = [1, 2, 3, 7];
    let mut it = xs.iter().cloned().zip(ys);
    assert_eq!(it.next(), Some((1, 1)));
    assert_eq!(it.next(), Some((2, 2)));
    assert_eq!(it.next_back(), Some((4, 7)));
    assert_eq!(it.next_back(), Some((3, 3)));
    assert_eq!(it.next(), None);
}

#[test]
#[cfg(panic = "unwind")]
/// Regresion test for #137255
/// A previous implementation of Zip TrustedRandomAccess specializations tried to do a lot of work
/// to preserve side-effects of equalizing the iterator lengths during backwards iteration.
/// This lead to several cases of unsoundness, twice due to being left in an inconsistent state
/// after panics.
/// The new implementation does not try as hard, but we still need panic-safety.
fn test_nested_zip_panic_safety() {
    use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
    use std::sync::atomic::{AtomicUsize, Ordering};

    let mut panic = true;
    // keeps track of how often element get visited, must be at most once each
    let witness = [8, 9, 10, 11, 12].map(|i| (i, AtomicUsize::new(0)));
    let a = witness.as_slice().iter().map(|e| {
        e.1.fetch_add(1, Ordering::Relaxed);
        if panic {
            panic = false;
            resume_unwind(Box::new(()))
        }
        e.0
    });
    // shorter than `a`, so `a` will get trimmed
    let b = [1, 2, 3, 4].as_slice().iter().copied();
    // shorter still, so `ab` will get trimmed.`
    let c = [5, 6, 7].as_slice().iter().copied();

    // This will panic during backwards trimming.
    let ab = zip(a, b);
    // This being Zip + TrustedRandomAccess means it will only call `next_back``
    // during trimming and otherwise do calls `__iterator_get_unchecked` on `ab`.
    let mut abc = zip(ab, c);

    assert_eq!(abc.len(), 3);
    // This will first trigger backwards trimming before it would normally obtain the
    // actual element if it weren't for the panic.
    // This used to corrupt the internal state of `abc`, which then lead to
    // TrustedRandomAccess safety contract violations in calls to  `ab`,
    // which ultimately lead to UB.
    catch_unwind(AssertUnwindSafe(|| abc.next_back())).ok();
    // check for sane outward behavior after the panic, which indicates a sane internal state.
    // Technically these outcomes are not required because a panic frees us from correctness obligations.
    assert_eq!(abc.len(), 2);
    assert_eq!(abc.next(), Some(((8, 1), 5)));
    assert_eq!(abc.next_back(), Some(((9, 2), 6)));
    for (i, (_, w)) in witness.iter().enumerate() {
        let v = w.load(Ordering::Relaxed);
        // required by TRA contract
        assert!(v <= 1, "expected idx {i} to be visited at most once, actual: {v}");
    }
    // Trimming panicked and should only run once, so this one won't be visited.
    // Implementation detail, but not trying to run it again is what keeps
    // things simple.
    assert_eq!(witness[3].1.load(Ordering::Relaxed), 0);
}

#[test]
fn test_issue_82282() {
    fn overflowed_zip(arr: &[i32]) -> impl Iterator<Item = (i32, &())> {
        static UNIT_EMPTY_ARR: [(); 0] = [];

        let mapped = arr.into_iter().map(|i| *i);
        let mut zipped = mapped.zip(UNIT_EMPTY_ARR.iter());
        zipped.next();
        zipped
    }

    let arr = [1, 2, 3];
    let zip = overflowed_zip(&arr).zip(overflowed_zip(&arr));

    assert_eq!(zip.size_hint(), (0, Some(0)));
    for _ in zip {
        panic!();
    }
}

#[test]
fn test_issue_82291() {
    use std::cell::Cell;

    let mut v1 = [()];
    let v2 = [()];

    let called = Cell::new(0);

    let mut zip = v1
        .iter_mut()
        .map(|r| {
            called.set(called.get() + 1);
            r
        })
        .zip(&v2);

    zip.next_back();
    assert_eq!(called.get(), 1);
    zip.next();
    assert_eq!(called.get(), 1);
}
