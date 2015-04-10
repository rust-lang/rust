// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 15523: derive(PartialOrd) should use the provided
// discriminant values for the derived ordering.
//
// This test is checking corner cases that arise when you have
// 64-bit values in the variants.

#[derive(PartialEq, PartialOrd)]
#[repr(u64)]
enum Eu64 {
    Pos2 = 2,
    PosMax = !0,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(i64)]
enum Ei64 {
    Pos2 = 2,
    Neg1 = -1,
    NegMin = 1 << 63,
    PosMax = !(1 << 63),
    Pos1 = 1,
}

fn main() {
    assert!(Eu64::Pos2 > Eu64::Pos1);
    assert!(Eu64::Pos2 < Eu64::PosMax);
    assert!(Eu64::Pos1 < Eu64::PosMax);


    assert!(Ei64::Pos2 > Ei64::Pos1);
    assert!(Ei64::Pos2 > Ei64::Neg1);
    assert!(Ei64::Pos1 > Ei64::Neg1);
    assert!(Ei64::Pos2 > Ei64::NegMin);
    assert!(Ei64::Pos1 > Ei64::NegMin);
    assert!(Ei64::Pos2 < Ei64::PosMax);
    assert!(Ei64::Pos1 < Ei64::PosMax);
}
