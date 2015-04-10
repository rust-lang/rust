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

#[derive(PartialEq, PartialOrd)]
enum E1 {
    Pos2 = 2,
    Neg1 = -1,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(u8)]
enum E2 {
    Pos2 = 2,
    PosMax = !0 as u8,
    Pos1 = 1,
}

#[derive(PartialEq, PartialOrd)]
#[repr(i8)]
enum E3 {
    Pos2 = 2,
    Neg1 = -1_i8,
    Pos1 = 1,
}

fn main() {
    assert!(E1::Pos2 > E1::Pos1);
    assert!(E1::Pos1 > E1::Neg1);
    assert!(E1::Pos2 > E1::Neg1);

    assert!(E2::Pos2 > E2::Pos1);
    assert!(E2::Pos1 < E2::PosMax);
    assert!(E2::Pos2 < E2::PosMax);

    assert!(E3::Pos2 > E3::Pos1);
    assert!(E3::Pos1 > E3::Neg1);
    assert!(E3::Pos2 > E3::Neg1);
}
