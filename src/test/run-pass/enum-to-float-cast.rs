// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that enum-to-float-casts do *signed* integer-to-float conversion.

enum E {
    L0 = -1,
    H0 = 1
}

enum F {
    L1 = 1,
    H1 = 0xFFFFFFFFFFFFFFFF
}

static C0: f32 = L0 as f32;
static C1: f32 = H1 as f32;

pub fn main() {
    let a = L0 as f32;
    let b = C0;
    let c = H1 as f32;
    let d = C1;
    assert_eq!(a, -1.0f32);
    assert_eq!(b, -1.0f32);
    assert_eq!(c, -1.0f32);
    assert_eq!(d, -1.0f32);
}
