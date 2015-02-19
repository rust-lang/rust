// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test range syntax.

fn foo() -> int { 42 }

// Test that range syntax works in return statements
fn return_range_to() -> ::std::ops::RangeTo<i32> { return ..1; }
fn return_full_range() -> ::std::ops::RangeFull { return ..; }

pub fn main() {
    let mut count = 0;
    for i in 0_usize..10 {
        assert!(i >= 0 && i < 10);
        count += i;
    }
    assert!(count == 45);

    let mut count = 0;
    let mut range = 0_usize..10;
    for i in range {
        assert!(i >= 0 && i < 10);
        count += i;
    }
    assert!(count == 45);

    let mut count = 0;
    let mut rf = 3_usize..;
    for i in rf.take(10) {
        assert!(i >= 3 && i < 13);
        count += i;
    }
    assert!(count == 75);

    let _ = 0_usize..4+4-3;
    let _ = 0..foo();

    let _ = ..42_usize;

    // Test we can use two different types with a common supertype.
    let x = &42;
    {
        let y = 42;
        let _ = x..&y;
    }
}
