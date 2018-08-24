use core::ops::{Range, RangeFull, RangeFrom, RangeTo, RangeInclusive};

// Test the Range structs without the syntactic sugar.

#[test]
fn test_range() {
    let r = Range { start: 2, end: 10 };
    let mut count = 0;
    for (i, ri) in r.enumerate() {
        assert!(ri == i + 2);
        assert!(ri >= 2 && ri < 10);
        count += 1;
    }
    assert!(count == 8);
}

#[test]
fn test_range_from() {
    let r = RangeFrom { start: 2 };
    let mut count = 0;
    for (i, ri) in r.take(10).enumerate() {
        assert!(ri == i + 2);
        assert!(ri >= 2 && ri < 12);
        count += 1;
    }
    assert!(count == 10);
}

#[test]
fn test_range_to() {
    // Not much to test.
    let _ = RangeTo { end: 42 };
}

#[test]
fn test_full_range() {
    // Not much to test.
    let _ = RangeFull;
}

#[test]
fn test_range_inclusive() {
    let mut r = RangeInclusive::new(1i8, 2);
    assert_eq!(r.next(), Some(1));
    assert_eq!(r.next(), Some(2));
    assert_eq!(r.next(), None);

    r = RangeInclusive::new(127i8, 127);
    assert_eq!(r.next(), Some(127));
    assert_eq!(r.next(), None);

    r = RangeInclusive::new(-128i8, -128);
    assert_eq!(r.next_back(), Some(-128));
    assert_eq!(r.next_back(), None);

    // degenerate
    r = RangeInclusive::new(1, -1);
    assert_eq!(r.size_hint(), (0, Some(0)));
    assert_eq!(r.next(), None);
}


#[test]
fn test_range_is_empty() {
    use core::f32::*;

    assert!(!(0.0 .. 10.0).is_empty());
    assert!( (-0.0 .. 0.0).is_empty());
    assert!( (10.0 .. 0.0).is_empty());

    assert!(!(NEG_INFINITY .. INFINITY).is_empty());
    assert!( (EPSILON .. NAN).is_empty());
    assert!( (NAN .. EPSILON).is_empty());
    assert!( (NAN .. NAN).is_empty());

    assert!(!(0.0 ..= 10.0).is_empty());
    assert!(!(-0.0 ..= 0.0).is_empty());
    assert!( (10.0 ..= 0.0).is_empty());

    assert!(!(NEG_INFINITY ..= INFINITY).is_empty());
    assert!( (EPSILON ..= NAN).is_empty());
    assert!( (NAN ..= EPSILON).is_empty());
    assert!( (NAN ..= NAN).is_empty());
}
