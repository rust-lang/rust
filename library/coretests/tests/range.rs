//! Various tests for the new-style range types

use core::cmp::Ordering;
use core::iter::Step;
use core::num::NonZero;
use core::ops::ControlFlow;
use core::panicking::panic;
use core::range::RangeInclusive;

#[test]
fn test_range_inclusive_to_exclusive_transform() {
    // The Debug format is *not* a stable guarantee, but is convenient for internal tests.
    let iter = RangeInclusive { start: '0', last: '9' }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter('0'..':')");

    let iter = RangeInclusive { start: 10, last: 100 }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(10..101)");
    let iter = RangeInclusive { start: 100, last: 100 }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(100..101)");
    let iter = RangeInclusive { start: 100, last: 10 }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(100..11)");
    let iter = RangeInclusive { start: 0, last: 255_u8 }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(0..=255)");
    let iter = RangeInclusive { start: 255, last: 255_u8 }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(255..=255)");
    let iter = RangeInclusive { start: 255_u8, last: 254 }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(255..255)");

    // Also check with a !Ord type...
    let iter = RangeInclusive { start: NotOrd::A(200), last: NotOrd::A(255) }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(A(200)..=A(255))");
    assert_eq!(iter.clone().next(), Some(NotOrd::A(200)));
    assert_eq!(iter.clone().next_back(), Some(NotOrd::A(255)));
    let iter = RangeInclusive { start: NotOrd::B(200), last: NotOrd::B(255) }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(B(200)..B(256))");
    assert_eq!(iter.clone().next(), Some(NotOrd::B(200)));
    assert_eq!(iter.clone().next_back(), Some(NotOrd::B(255)));
    // ...particularly for these cases where neither start ≤ last nor start ≥ last.
    let iter = RangeInclusive { start: NotOrd::A(200), last: NotOrd::B(255) }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(A(200)..B(256))");
    assert_eq!(iter.clone().next(), None);
    assert_eq!(iter.clone().next_back(), None);
    let iter = RangeInclusive { start: NotOrd::B(200), last: NotOrd::A(255) }.into_iter();
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(B(200)..A(255))");
    assert_eq!(iter.clone().next(), None);
    assert_eq!(iter.clone().next_back(), None);
}

#[test]
fn test_range_inclusive_iter_exclusive_inner() {
    let mut iter = RangeInclusive::<u8> { start: 10, last: 12 }.into_iter();
    assert_eq!(iter.next(), Some(10));
    assert_eq!(iter.next(), Some(11));
    assert_eq!(iter.next(), Some(12));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    let mut iter = RangeInclusive::<u8> { start: 10, last: 12 }.into_iter();
    assert_eq!(iter.next_back(), Some(12));
    assert_eq!(iter.next_back(), Some(11));
    assert_eq!(iter.next_back(), Some(10));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next_back(), None);
}

#[test]
fn test_range_inclusive_iter_inclusive_inner() {
    let mut iter = RangeInclusive::<u8> { start: 252, last: 255 }.into_iter();
    assert_eq!(iter.next(), Some(252));
    assert_eq!(iter.next(), Some(253));
    assert_eq!(iter.next(), Some(254));
    assert_eq!(iter.next(), Some(255));
    assert_eq!(iter.next(), None);
    assert_eq!(iter.next(), None);

    let mut iter = RangeInclusive::<u8> { start: 252, last: 255 }.into_iter();
    assert_eq!(iter.next_back(), Some(255));
    assert_eq!(iter.next_back(), Some(254));
    assert_eq!(iter.next_back(), Some(253));
    assert_eq!(iter.next_back(), Some(252));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next_back(), None);

    let mut iter = RangeInclusive::<u8> { start: 253, last: 255 }.into_iter();
    assert_eq!(iter.next(), Some(253));
    assert_eq!(iter.next_back(), Some(255));
    assert_eq!(iter.next(), Some(254));
    assert_eq!(iter.next_back(), None);
    assert_eq!(iter.next(), None);
}

#[test]
fn test_range_inclusive_iter_folds() {
    let iter = RangeInclusive::<u8> { start: 53, last: 55 }.into_iter();
    let mut vec = Vec::new();
    let count = iter.fold(0, |i, x| {
        vec.push((i, x));
        i + 1
    });
    assert_eq!(count, 3);
    assert_eq!(vec, [(0, 53), (1, 54), (2, 55)]);

    let iter = RangeInclusive::<u8> { start: 253, last: 255 }.into_iter();
    let mut vec = Vec::new();
    let count = iter.fold(0, |i, x| {
        vec.push((i, x));
        i + 1
    });
    assert_eq!(count, 3);
    assert_eq!(vec, [(0, 253), (1, 254), (2, 255)]);

    let iter = RangeInclusive::<u8> { start: 53, last: 55 }.into_iter();
    let mut vec = Vec::new();
    let count = iter.rfold(0, |i, x| {
        vec.push((i, x));
        i + 1
    });
    assert_eq!(count, 3);
    assert_eq!(vec, [(0, 55), (1, 54), (2, 53)]);

    let iter = RangeInclusive::<u8> { start: 253, last: 255 }.into_iter();
    let mut vec = Vec::new();
    let count = iter.rfold(0, |i, x| {
        vec.push((i, x));
        i + 1
    });
    assert_eq!(count, 3);
    assert_eq!(vec, [(0, 255), (1, 254), (2, 253)]);
}

#[test]
fn test_range_inclusive_iter_try_resumption() {
    let mut iter = RangeInclusive::<u8> { start: 53, last: 55 }.into_iter();
    let mut n = || iter.try_for_each(ControlFlow::Break).break_value();
    assert_eq!(n(), Some(53));
    assert_eq!(n(), Some(54));
    assert_eq!(n(), Some(55));
    assert_eq!(n(), None);
    assert_eq!(n(), None);

    let mut iter = RangeInclusive::<u8> { start: 253, last: 255 }.into_iter();
    let mut n = || iter.try_for_each(ControlFlow::Break).break_value();
    assert_eq!(n(), Some(253));
    assert_eq!(n(), Some(254));
    assert_eq!(n(), Some(255));
    assert_eq!(n(), None);
    assert_eq!(n(), None);

    let mut iter = RangeInclusive::<u8> { start: 53, last: 55 }.into_iter().rev();
    let mut n = || iter.try_for_each(ControlFlow::Break).break_value();
    assert_eq!(n(), Some(55));
    assert_eq!(n(), Some(54));
    assert_eq!(n(), Some(53));
    assert_eq!(n(), None);
    assert_eq!(n(), None);

    let mut iter = RangeInclusive::<u8> { start: 253, last: 255 }.into_iter().rev();
    let mut n = || iter.try_for_each(ControlFlow::Break).break_value();
    assert_eq!(n(), Some(255));
    assert_eq!(n(), Some(254));
    assert_eq!(n(), Some(253));
    assert_eq!(n(), None);
    assert_eq!(n(), None);
}

#[test]
fn test_range_inclusive_iter_advance() {
    // The Debug format is *not* a stable guarantee, but is convenient for internal tests.

    let in_middle = || RangeInclusive::<i8> { start: 10, last: 12 }.into_iter();
    assert_eq!(format!("{:?}", in_middle()), "RangeInclusiveIter(10..13)");
    let at_end = || RangeInclusive::<i8> { start: 125, last: 127 }.into_iter();
    assert_eq!(format!("{:?}", at_end()), "RangeInclusiveIter(125..=127)");

    let mut iter = in_middle();
    assert_eq!(iter.advance_by(2), Ok(()));
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(12..13)");
    assert_eq!(iter.advance_by(2), Err(NonZero::new(1).unwrap()));
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(13..13)");

    let mut iter = at_end();
    assert_eq!(iter.advance_by(2), Ok(()));
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(127..=127)");
    assert_eq!(iter.advance_by(2), Err(NonZero::new(1).unwrap()));
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(127..127)");

    let mut iter = in_middle();
    assert_eq!(iter.advance_back_by(2), Ok(()));
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(10..11)");
    assert_eq!(iter.advance_back_by(2), Err(NonZero::new(1).unwrap()));
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(10..10)");

    let mut iter = at_end();
    assert_eq!(iter.advance_back_by(2), Ok(()));
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(125..126)");
    assert_eq!(iter.advance_back_by(2), Err(NonZero::new(1).unwrap()));
    assert_eq!(format!("{iter:?}"), "RangeInclusiveIter(125..125)");
}

#[test]
fn test_range_inclusive_iter_empty_and_remainder() {
    let values = [u8::MIN, u8::MIN + 1, 6, 7, u8::MAX - 1, u8::MAX];
    for start in values {
        for last in values {
            let iter = RangeInclusive { start, last }.into_iter();
            let non_empty = start <= last;
            let expected_remainder = non_empty.then_some(RangeInclusive { start, last });
            assert_eq!(iter.is_empty(), !non_empty);
            assert_eq!(iter.remainder(), expected_remainder);
        }
    }
}

/// A type that's a valid `Step` but isn't `Ord`
#[derive_const(Clone, PartialEq)]
#[derive(Debug)]
enum NotOrd {
    A(u8),
    B(usize),
}
const impl core::cmp::PartialOrd for NotOrd {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        match (self, other) {
            (NotOrd::A(left), NotOrd::A(right)) => Some(Ord::cmp(left, right)),
            (NotOrd::B(left), NotOrd::B(right)) => Some(Ord::cmp(left, right)),
            _ => None,
        }
    }
}
const impl Step for NotOrd {
    fn steps_between(_start: &Self, _end: &Self) -> (usize, Option<usize>) {
        // I guess?
        (0, None)
    }
    fn forward_checked(start: Self, count: usize) -> Option<Self> {
        match start {
            NotOrd::A(v) => {
                let Ok(count) = count.try_into() else { return None };
                v.checked_add(count).map(NotOrd::A)
            }
            NotOrd::B(v) => v.checked_add(count).map(NotOrd::B),
        }
    }
    fn forward_overflowing(_start: Self, _count: usize) -> (Self, bool) {
        panic("todo")
    }
    fn backward_checked(start: Self, count: usize) -> Option<Self> {
        match start {
            NotOrd::A(v) => {
                let Ok(count) = count.try_into() else { return None };
                v.checked_sub(count).map(NotOrd::A)
            }
            NotOrd::B(v) => v.checked_sub(count).map(NotOrd::B),
        }
    }
    fn backward_overflowing(_start: Self, _count: usize) -> (Self, bool) {
        panic("todo")
    }
}
