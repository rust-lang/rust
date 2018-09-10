// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass

#![feature(const_int_sign)]

const NEGATIVE_A: bool = (-10i32).is_negative();
const NEGATIVE_B: bool = 10i32.is_negative();
const POSITIVE_A: bool= (-10i32).is_positive();
const POSITIVE_B: bool= 10i32.is_positive();

fn main() {
    assert!(NEGATIVE_A);
    assert!(!NEGATIVE_B);
    assert!(!POSITIVE_A);
    assert!(POSITIVE_B);
}
