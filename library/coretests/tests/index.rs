use core::index::Clamp;
use core::range;
use core::slice::SliceIndex;

macro_rules! test_clamp {
    ($range:expr, $(($slice:expr, $other:expr)),+) => {
        $(
            assert_eq!(Clamp($range.clone()).get(&$slice as &[_]), $other.get(&$slice as &[_]));
            assert_eq!(Clamp($range.clone()).get_mut(&mut $slice as &mut [_]), $other.get_mut(&mut $slice as &mut [_]));
            unsafe {
                assert_eq!(&*Clamp($range.clone()).get_unchecked(&$slice as &[_]), &*$other.get_unchecked(&$slice as &[_]));
                assert_eq!(&*Clamp($range.clone()).get_unchecked_mut(&mut $slice as &mut [_]), &*$other.get_unchecked_mut(&mut $slice as &mut [_]));
            }
            assert_eq!(Clamp($range.clone()).index(&$slice as &[_]), $other.index(&$slice as &[_]));
            assert_eq!(Clamp($range.clone()).index_mut(&mut $slice as &mut [_]), $other.index_mut(&mut $slice as &mut [_]));
        )+
    };
}

#[test]
fn test_clamp_usize() {
    test_clamp!(2, ([0, 1], 1), ([0, 1, 2], 2));
}

#[test]
fn test_clamp_range_range() {
    test_clamp!(range::Range::from(1..4), ([0, 1], 1..2), ([0, 1, 2, 3, 4], 1..4), ([0], 1..1));
}

#[test]
fn test_clamp_ops_range() {
    test_clamp!(1..4, ([0, 1], 1..2), ([0, 1, 2, 3, 4], 1..4), ([0], 1..1));
}

#[test]
fn test_clamp_range_range_inclusive() {
    test_clamp!(
        range::RangeInclusive::from(1..=3),
        ([0, 1], 1..=1),
        ([0, 1, 2, 3, 4], 1..=3),
        ([0], 0..=0)
    );
}

#[test]
fn test_clamp_ops_range_inclusive() {
    test_clamp!(1..=3, ([0, 1], 1..=1), ([0, 1, 2, 3, 4], 1..=3), ([0], 0..=0));
}

#[test]
fn test_clamp_range_range_from() {
    test_clamp!(range::RangeFrom::from(1..), ([0, 1], 1..), ([0, 1, 2, 3, 4], 1..), ([0], 1..));
}

#[test]
fn test_clamp_ops_range_from() {
    test_clamp!(1.., ([0, 1], 1..), ([0, 1, 2, 3, 4], 1..), ([0], 1..));
}

#[test]
fn test_clamp_range_to() {
    test_clamp!(..4, ([0, 1], ..2), ([0, 1, 2, 3, 4], ..4), ([0], ..1));
}

#[test]
fn test_clamp_range_range_to_inclusive() {
    test_clamp!(
        range::RangeToInclusive::from(..=4),
        ([0, 1], ..=1),
        ([0, 1, 2, 3, 4], ..=4),
        ([0], ..=0)
    );
}

#[test]
fn test_clamp_ops_range_to_inclusive() {
    test_clamp!(..=4, ([0, 1], ..=1), ([0, 1, 2, 3, 4], ..=4), ([0], ..=0));
}

#[test]
fn test_clamp_range_full() {
    test_clamp!(.., ([0, 1], ..), ([0, 1, 2, 3, 4], ..), ([0], ..));
}
