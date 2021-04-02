use core::iter::*;

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
    let a = xs.iter().cloned();
    let b = ys.iter().cloned();
    let mut it = a.zip(b);
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
