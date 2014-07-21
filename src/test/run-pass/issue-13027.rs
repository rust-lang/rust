// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//
// ignore-lexer-test FIXME #15877

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
    misc();
}

fn lit_shadow_range() {
    assert_eq!(2i, match 1i {
        1 if false => 1i,
        1..2 => 2,
        _ => 3
    });

    let x = 0i;
    assert_eq!(2i, match x+1 {
        0 => 0i,
        1 if false => 1,
        1..2 => 2,
        _ => 3
    });

    assert_eq!(2i, match val() {
        1 if false => 1i,
        1..2 => 2,
        _ => 3
    });

    assert_eq!(2i, match CONST {
        0 => 0i,
        1 if false => 1,
        1..2 => 2,
        _ => 3
    });

    // value is out of the range of second arm, should match wildcard pattern
    assert_eq!(3i, match 3i {
        1 if false => 1i,
        1..2 => 2,
        _ => 3
    });
}

fn range_shadow_lit() {
    assert_eq!(2i, match 1i {
        1..2 if false => 1i,
        1 => 2,
        _ => 3
    });

    let x = 0i;
    assert_eq!(2i, match x+1 {
        0 => 0i,
        1..2 if false => 1,
        1 => 2,
        _ => 3
    });

    assert_eq!(2i, match val() {
        1..2 if false => 1i,
        1 => 2,
        _ => 3
    });

    assert_eq!(2i, match CONST {
        0 => 0i,
        1..2 if false => 1,
        1 => 2,
        _ => 3
    });

    // ditto
    assert_eq!(3i, match 3i {
        1..2 if false => 1i,
        1 => 2,
        _ => 3
    });
}

fn range_shadow_range() {
    assert_eq!(2i, match 1i {
        0..2 if false => 1i,
        1..3 => 2,
        _ => 3,
    });

    let x = 0i;
    assert_eq!(2i, match x+1 {
        100 => 0,
        0..2 if false => 1,
        1..3 => 2,
        _ => 3,
    });

    assert_eq!(2i, match val() {
        0..2 if false => 1,
        1..3 => 2,
        _ => 3,
    });

    assert_eq!(2i, match CONST {
        100 => 0,
        0..2 if false => 1,
        1..3 => 2,
        _ => 3,
    });

    // ditto
    assert_eq!(3i, match 5i {
        0..2 if false => 1i,
        1..3 => 2,
        _ => 3,
    });
}

fn multi_pats_shadow_lit() {
    assert_eq!(2i, match 1i {
        100 => 0i,
        0 | 1..10 if false => 1,
        1 => 2,
        _ => 3,
    });
}

fn multi_pats_shadow_range() {
    assert_eq!(2i, match 1i {
        100 => 0i,
        0 | 1..10 if false => 1,
        1..3 => 2,
        _ => 3,
    });
}

fn lit_shadow_multi_pats() {
    assert_eq!(2i, match 1i {
        100 => 0i,
        1 if false => 1,
        0 | 1..10 => 2,
        _ => 3,
    });
}

fn range_shadow_multi_pats() {
    assert_eq!(2i, match 1i {
        100 => 0i,
        1..3 if false => 1,
        0 | 1..10 => 2,
        _ => 3,
    });
}

fn misc() {
    enum Foo {
        Bar(uint, bool)
    }
    // This test basically mimics how trace_macros! macro is implemented,
    // which is a rare combination of vector patterns, multiple wild-card
    // patterns and guard functions.
    let r = match [Bar(0, false)].as_slice() {
        [Bar(_, pred)] if pred => 1i,
        [Bar(_, pred)] if !pred => 2i,
        _ => 0i,
    };
    assert_eq!(2i, r);
}
