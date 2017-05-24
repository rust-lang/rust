// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

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
    let mut r = RangeInclusive { start: 1i8, end: 2 };
    assert_eq!(r.next(), Some(1));
    assert_eq!(r.next(), Some(2));
    assert_eq!(r.next(), None);

    r = RangeInclusive { start: 127i8, end: 127 };
    assert_eq!(r.next(), Some(127));
    assert_eq!(r.next(), None);

    r = RangeInclusive { start: -128i8, end: -128 };
    assert_eq!(r.next_back(), Some(-128));
    assert_eq!(r.next_back(), None);

    // degenerate
    r = RangeInclusive { start: 1, end: -1 };
    assert_eq!(r.size_hint(), (0, Some(0)));
    assert_eq!(r.next(), None);
}