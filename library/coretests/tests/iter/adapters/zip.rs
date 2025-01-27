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
    iter.next();
    assert_eq!(iter.next_back(), None);
    assert_eq!(a, vec![1, 2, 3, 4, 6, 5]);
    assert_eq!(b, vec![200, 300, 400]);
}

#[test]
fn test_zip_cloned_sideffectful() {
    let xs = [CountClone::new(), CountClone::new(), CountClone::new(), CountClone::new()];
    let ys = [CountClone::new(), CountClone::new()];

    for _ in xs.iter().cloned().zip(ys.iter().cloned()) {}

    assert_eq!(&xs, &[1, 1, 1, 0][..]);
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

    assert_eq!(&xs, &[1, 1, 1, 1, 1, 0]);
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
        (&mut it).take(5).count();
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
    iter.next();
    assert_eq!(iter.nth_back(0), None);
    assert_eq!(a, vec![1, 2, 3, 4, 6, 5]);
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
#[cfg(panic = "unwind")]
fn test_zip_trusted_random_access_next_back_drop() {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    let mut counter = 0;

    let it = [42].iter().map(|e| {
        let c = counter;
        counter += 1;
        if c == 0 {
            panic!("bomb");
        }

        e
    });
    let it2 = [(); 0].iter();
    let mut zip = it.zip(it2);
    catch_unwind(AssertUnwindSafe(|| {
        zip.next_back();
    }))
    .unwrap_err();
    assert!(zip.next().is_none());
    assert_eq!(counter, 1);
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
