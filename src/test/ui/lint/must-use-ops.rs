// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue #50124 - Test warning for unused operator expressions

// compile-pass

#![warn(unused_must_use)]

fn main() {
    let val = 1;
    let val_pointer = &val;

// Comparison Operators
    val == 1;
    val < 1;
    val <= 1;
    val != 1;
    val >= 1;
    val > 1;

// Arithmetic Operators
    val + 2;
    val - 2;
    val / 2;
    val * 2;
    val % 2;

// Logical Operators
    true && true;
    false || true;

// Bitwise Operators
    5 ^ val;
    5 & val;
    5 | val;
    5 << val;
    5 >> val;

// Unary Operators
    !val;
    -val;
    *val_pointer;
}
