// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z force-overflow-checks=off

// Test that with MIR trans, overflow checks can be
// turned off, even when they're from core::ops::*.

use std::ops::*;

fn main() {
    assert_eq!(i8::neg(-0x80), -0x80);

    assert_eq!(u8::add(0xff, 1), 0_u8);
    assert_eq!(u8::sub(0, 1), 0xff_u8);
    assert_eq!(u8::mul(0xff, 2), 0xfe_u8);
    assert_eq!(u8::shl(1, 9), 2_u8);
    assert_eq!(u8::shr(2, 9), 1_u8);
}
