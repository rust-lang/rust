mod control_flow;
mod from_residual;

use core::ops::{
    Bound, Deref, DerefMut, OneSidedRange, OneSidedRangeBound, Range, RangeBounds, RangeFrom,
    RangeFull, RangeInclusive, RangeTo, RangeToInclusive,
};

// Test the Range structs and syntax.

#[test]
fn test_range() {
    let r = Range { start: 2, end: 10 };
    let mut count = 0;
    for (i, ri) in r.enumerate() {
        assert_eq!(ri, i + 2);
        assert!(ri >= 2 && ri < 10);
        count += 1;
    }
    assert_eq!(count, 8);
}

#[test]
fn test_range_from() {
    let r = RangeFrom { start: 2 };
    let mut count = 0;
    for (i, ri) in r.take(10).enumerate() {
        assert_eq!(ri, i + 2);
        assert!(ri >= 2 && ri < 12);
        count += 1;
    }
    assert_eq!(count, 10);
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
fn test_range_to_inclusive() {
    // Not much to test.
    let _ = RangeToInclusive { end: 42 };
}

#[test]
fn test_range_contains() {
    assert!(!(1u32..5).contains(&0u32));
    assert!((1u32..5).contains(&1u32));
    assert!((1u32..5).contains(&4u32));
    assert!(!(1u32..5).contains(&5u32));
    assert!(!(1u32..5).contains(&6u32));
}

#[test]
fn test_range_to_contains() {
    assert!(!(1u32..=5).contains(&0));
    assert!((1u32..=5).contains(&1));
    assert!((1u32..=5).contains(&4));
    assert!((1u32..=5).contains(&5));
    assert!(!(1u32..=5).contains(&6));
}

// This test covers `RangeBounds::contains` when the start is excluded,
// which cannot be directly expressed by Rust's built-in range syntax.
#[test]
fn test_range_bounds_contains() {
    let r = (Bound::Excluded(1u32), Bound::Included(5u32));
    assert!(!r.contains(&0));
    assert!(!r.contains(&1));
    assert!(r.contains(&3));
    assert!(r.contains(&5));
    assert!(!r.contains(&6));
}

#[test]
fn test_range_is_empty() {
    assert!(!(0.0..10.0).is_empty());
    assert!((-0.0..0.0).is_empty());
    assert!((10.0..0.0).is_empty());

    assert!(!(f32::NEG_INFINITY..f32::INFINITY).is_empty());
    assert!((f32::EPSILON..f32::NAN).is_empty());
    assert!((f32::NAN..f32::EPSILON).is_empty());
    assert!((f32::NAN..f32::NAN).is_empty());

    assert!(!(0.0..=10.0).is_empty());
    assert!(!(-0.0..=0.0).is_empty());
    assert!((10.0..=0.0).is_empty());

    assert!(!(f32::NEG_INFINITY..=f32::INFINITY).is_empty());
    assert!((f32::EPSILON..=f32::NAN).is_empty());
    assert!((f32::NAN..=f32::EPSILON).is_empty());
    assert!((f32::NAN..=f32::NAN).is_empty());
}

#[test]
fn test_range_inclusive_end_bound() {
    let mut r = 1u32..=1;
    r.next().unwrap();
    assert!(!r.contains(&1));
}

#[test]
fn test_range_bounds() {
    let r = (Bound::Included(1u32), Bound::Excluded(5u32));
    assert!(!r.contains(&0));
    assert!(r.contains(&1));
    assert!(r.contains(&3));
    assert!(!r.contains(&5));
    assert!(!r.contains(&6));

    let r = (Bound::<u32>::Unbounded, Bound::Unbounded);
    assert!(r.contains(&0));
    assert!(r.contains(&u32::MAX));
}

#[test]
fn test_one_sided_range_bound() {
    assert!(matches!((..1u32).bound(), (OneSidedRangeBound::End, 1)));
    assert!(matches!((1u32..).bound(), (OneSidedRangeBound::StartInclusive, 1)));
    assert!(matches!((..=1u32).bound(), (OneSidedRangeBound::EndInclusive, 1)));
}

#[test]
fn test_bound_cloned_unbounded() {
    assert_eq!(Bound::<&u32>::Unbounded.cloned(), Bound::Unbounded);
}

#[test]
fn test_bound_cloned_included() {
    assert_eq!(Bound::Included(&3).cloned(), Bound::Included(3));
}

#[test]
fn test_bound_cloned_excluded() {
    assert_eq!(Bound::Excluded(&3).cloned(), Bound::Excluded(3));
}

#[test]
#[allow(unused_comparisons)]
#[allow(unused_mut)]
fn test_range_syntax() {
    let mut count = 0;
    for i in 0_usize..10 {
        assert!(i >= 0 && i < 10);
        count += i;
    }
    assert_eq!(count, 45);

    let mut count = 0;
    let mut range = 0_usize..10;
    for i in range {
        assert!(i >= 0 && i < 10);
        count += i;
    }
    assert_eq!(count, 45);

    let mut count = 0;
    let mut rf = 3_usize..;
    for i in rf.take(10) {
        assert!(i >= 3 && i < 13);
        count += i;
    }
    assert_eq!(count, 75);

    let _ = 0_usize..4 + 4 - 3;

    fn foo() -> isize {
        42
    }
    let _ = 0..foo();

    let _ = { &42..&100 }; // references to literals are OK
    let _ = ..42_usize;

    // Test we can use two different types with a common supertype.
    let x = &42;
    {
        let y = 42;
        let _ = x..&y;
    }
}

#[test]
#[allow(dead_code)]
fn test_range_syntax_in_return_statement() {
    fn return_range_to() -> RangeTo<i32> {
        return ..1;
    }
    fn return_full_range() -> RangeFull {
        return ..;
    }
    // Not much to test.
}

#[test]
fn range_structural_match() {
    // test that all range types can be structurally matched upon

    const RANGE: Range<usize> = 0..1000;
    match RANGE {
        RANGE => {}
        _ => unreachable!(),
    }

    const RANGE_FROM: RangeFrom<usize> = 0..;
    match RANGE_FROM {
        RANGE_FROM => {}
        _ => unreachable!(),
    }

    const RANGE_FULL: RangeFull = ..;
    match RANGE_FULL {
        RANGE_FULL => {}
    }

    const RANGE_INCLUSIVE: RangeInclusive<usize> = 0..=999;
    match RANGE_INCLUSIVE {
        RANGE_INCLUSIVE => {}
        _ => unreachable!(),
    }

    const RANGE_TO: RangeTo<usize> = ..1000;
    match RANGE_TO {
        RANGE_TO => {}
        _ => unreachable!(),
    }

    const RANGE_TO_INCLUSIVE: RangeToInclusive<usize> = ..=999;
    match RANGE_TO_INCLUSIVE {
        RANGE_TO_INCLUSIVE => {}
        _ => unreachable!(),
    }
}

// Test Deref implementations

#[test]
fn deref_mut_on_ref() {
    // Test that `&mut T` implements `DerefMut<T>`

    fn inc<T: Deref<Target = isize> + DerefMut>(mut t: T) {
        *t += 1;
    }

    let mut x: isize = 5;
    inc(&mut x);
    assert_eq!(x, 6);
}

#[test]
fn deref_on_ref() {
    // Test that `&T` and `&mut T` implement `Deref<T>`

    fn deref<U: Copy, T: Deref<Target = U>>(t: T) -> U {
        *t
    }

    let x: isize = 3;
    let y = deref(&x);
    assert_eq!(y, 3);

    let mut x: isize = 4;
    let y = deref(&mut x);
    assert_eq!(y, 4);
}

#[test]
#[allow(unreachable_code)]
fn test_not_never() {
    if !return () {}
}

#[test]
fn test_fmt() {
    let mut r = 1..=1;
    assert_eq!(format!("{:?}", r), "1..=1");
    r.next().unwrap();
    assert_eq!(format!("{:?}", r), "1..=1 (exhausted)");

    assert_eq!(format!("{:?}", 1..1), "1..1");
    assert_eq!(format!("{:?}", 1..), "1..");
    assert_eq!(format!("{:?}", ..1), "..1");
    assert_eq!(format!("{:?}", ..=1), "..=1");
    assert_eq!(format!("{:?}", ..), "..");
}
