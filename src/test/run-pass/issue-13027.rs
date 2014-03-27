// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that match expression handles overlapped literal and range
// properly in the presence of guard function.

fn val() -> uint { 1 }

static CONST: uint = 1;

pub fn main() {
    lit_shadow_range();
    range_shadow_lit();
    range_shadow_range();
    multi_pats_shadow_lit();
    multi_pats_shadow_range();
    lit_shadow_multi_pats();
    range_shadow_multi_pats();
}

fn lit_shadow_range() {
    assert_eq!(2, match 1 {
        1 if false => 1,
        1..2 => 2,
        _ => 3
    });

    let x = 0;
    assert_eq!(2, match x+1 {
        0 => 0,
        1 if false => 1,
        1..2 => 2,
        _ => 3
    });

    assert_eq!(2, match val() {
        1 if false => 1,
        1..2 => 2,
        _ => 3
    });

    assert_eq!(2, match CONST {
        0 => 0,
        1 if false => 1,
        1..2 => 2,
        _ => 3
    });

    // value is out of the range of second arm, should match wildcard pattern
    assert_eq!(3, match 3 {
        1 if false => 1,
        1..2 => 2,
        _ => 3
    });
}

fn range_shadow_lit() {
    assert_eq!(2, match 1 {
        1..2 if false => 1,
        1 => 2,
        _ => 3
    });

    let x = 0;
    assert_eq!(2, match x+1 {
        0 => 0,
        1..2 if false => 1,
        1 => 2,
        _ => 3
    });

    assert_eq!(2, match val() {
        1..2 if false => 1,
        1 => 2,
        _ => 3
    });

    assert_eq!(2, match CONST {
        0 => 0,
        1..2 if false => 1,
        1 => 2,
        _ => 3
    });

    // ditto
    assert_eq!(3, match 3 {
        1..2 if false => 1,
        1 => 2,
        _ => 3
    });
}

fn range_shadow_range() {
    assert_eq!(2, match 1 {
        0..2 if false => 1,
        1..3 => 2,
        _ => 3,
    });

    let x = 0;
    assert_eq!(2, match x+1 {
        100 => 0,
        0..2 if false => 1,
        1..3 => 2,
        _ => 3,
    });

    assert_eq!(2, match val() {
        0..2 if false => 1,
        1..3 => 2,
        _ => 3,
    });

    assert_eq!(2, match CONST {
        100 => 0,
        0..2 if false => 1,
        1..3 => 2,
        _ => 3,
    });

    // ditto
    assert_eq!(3, match 5 {
        0..2 if false => 1,
        1..3 => 2,
        _ => 3,
    });
}

fn multi_pats_shadow_lit() {
    assert_eq!(2, match 1 {
        100 => 0,
        0 | 1..10 if false => 1,
        1 => 2,
        _ => 3,
    });
}

fn multi_pats_shadow_range() {
    assert_eq!(2, match 1 {
        100 => 0,
        0 | 1..10 if false => 1,
        1..3 => 2,
        _ => 3,
    });
}

fn lit_shadow_multi_pats() {
    assert_eq!(2, match 1 {
        100 => 0,
        1 if false => 1,
        0 | 1..10 => 2,
        _ => 3,
    });
}

fn range_shadow_multi_pats() {
    assert_eq!(2, match 1 {
        100 => 0,
        1..3 if false => 1,
        0 | 1..10 => 2,
        _ => 3,
    });
}
