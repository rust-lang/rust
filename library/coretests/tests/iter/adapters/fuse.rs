use core::iter::*;

#[test]
fn test_fuse_fused_nth() {
    let xs = [0, 1, 2];
    let mut it = xs.iter().fuse();

    assert_eq!(it.len(), 3);
    assert_eq!(it.nth(2), Some(&2));
    assert_eq!(it.len(), 0);
    assert_eq!(it.nth(2), None);
    assert_eq!(it.len(), 0);
}

#[test]
fn test_fuse_fused_nth_back() {
    let xs = [0, 1, 2];
    let mut it = xs.iter().fuse();

    assert_eq!(it.len(), 3);
    assert_eq!(it.nth_back(2), Some(&0));
    assert_eq!(it.len(), 0);
    assert_eq!(it.nth_back(2), None);
    assert_eq!(it.len(), 0);
}

#[test]
fn test_fuse_fused_advance() {
    let xs = [0, 1, 2, 3];
    let mut it = xs.iter().fuse();

    assert_eq!(it.len(), 4);
    assert_eq!(it.advance_by(2), Ok(()));
    assert_eq!(it.next(), Some(&2));
    assert_eq!(it.len(), 1);
    assert_eq!(it.advance_by(2).map_err(|it| it.get()), Err(1));
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
}

#[test]
fn test_fuse_fused_advance_back() {
    let xs = [0, 1, 2, 3];
    let mut it = xs.iter().fuse();

    assert_eq!(it.len(), 4);
    assert_eq!(it.advance_back_by(2), Ok(()));
    assert_eq!(it.next_back(), Some(&1));
    assert_eq!(it.len(), 1);
    assert_eq!(it.advance_back_by(2).map_err(|it| it.get()), Err(1));
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
}

#[test]
fn test_fuse_unfused_nth() {
    let mut xs = [0, 1, 2].into_iter().map(Some).chain([None]).chain([3, 4, 5].map(Some));
    let mut it = core::iter::from_fn(|| xs.next().flatten()).fuse();

    assert_eq!(it.nth(1), Some(1));
    assert_eq!(it.nth(1), None);
    assert_eq!(xs.next(), Some(Some(3)));
}

#[test]
fn test_fuse_unfused_advance() {
    let mut xs = [0, 1, 2].into_iter().map(Some).chain([None]).chain([3, 4, 5].map(Some));
    let mut it = core::iter::from_fn(|| xs.next().flatten()).fuse();

    assert_eq!(it.advance_by(1), Ok(()));
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.advance_by(2).map_err(|it| it.get()), Err(1));
    assert_eq!(it.next(), None);
    assert_eq!(xs.next(), Some(Some(3)));
}

#[test]
fn test_fuse_last() {
    let xs = [0, 1, 2];
    let it = xs.iter().fuse();

    assert_eq!(it.len(), 3);
    assert_eq!(it.last(), Some(&2));
}

#[test]
fn test_fuse_count() {
    let xs = [0, 1, 2];
    let it = xs.iter().fuse();

    assert_eq!(it.len(), 3);
    assert_eq!(it.count(), 3);
    // Can't check len now because count consumes.
}

#[test]
fn test_fuse_fold() {
    let xs = [0, 1, 2];
    let it = xs.iter(); // `FusedIterator`
    let i = it.fuse().fold(0, |i, &x| {
        assert_eq!(x, xs[i]);
        i + 1
    });
    assert_eq!(i, xs.len());

    let it = xs.iter(); // `FusedIterator`
    let i = it.fuse().rfold(xs.len(), |i, &x| {
        assert_eq!(x, xs[i - 1]);
        i - 1
    });
    assert_eq!(i, 0);

    let it = xs.iter().scan((), |_, &x| Some(x)); // `!FusedIterator`
    let i = it.fuse().fold(0, |i, x| {
        assert_eq!(x, xs[i]);
        i + 1
    });
    assert_eq!(i, xs.len());
}

#[test]
fn test_fuse() {
    let mut it = (0..3).fuse();
    assert_eq!(it.len(), 3);
    assert_eq!(it.next(), Some(0));
    assert_eq!(it.len(), 2);
    assert_eq!(it.next(), Some(1));
    assert_eq!(it.len(), 1);
    assert_eq!(it.next(), Some(2));
    assert_eq!(it.len(), 0);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
    assert_eq!(it.next(), None);
    assert_eq!(it.len(), 0);
}
