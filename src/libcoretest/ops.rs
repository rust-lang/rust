// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use test::Bencher;
use core::ops::{Range, FullRange, RangeFrom, RangeTo};

// Overhead of dtors

struct HasDtor {
    _x: int
}

impl Drop for HasDtor {
    fn drop(&mut self) {
    }
}

#[bench]
fn alloc_obj_with_dtor(b: &mut Bencher) {
    b.iter(|| {
        HasDtor { _x : 10 };
    })
}

// Test the Range structs without the syntactic sugar.

#[test]
fn test_range() {
    let r = Range { start: 2u, end: 10 };
    let mut count = 0u;
    for (i, ri) in r.enumerate() {
        assert!(ri == i + 2);
        assert!(ri >= 2u && ri < 10u);
        count += 1;
    }
    assert!(count == 8);
}

#[test]
fn test_range_from() {
    let r = RangeFrom { start: 2u };
    let mut count = 0u;
    for (i, ri) in r.take(10).enumerate() {
        assert!(ri == i + 2);
        assert!(ri >= 2u && ri < 12u);
        count += 1;
    }
    assert!(count == 10);
}

#[test]
fn test_range_to() {
    // Not much to test.
    let _ = RangeTo { end: 42u };
}

#[test]
fn test_full_range() {
    // Not much to test.
    let _ = FullRange;
}
