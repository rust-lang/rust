// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that enum-to-float casts are disallowed.

enum E {
    L0 = -1,
    H0 = 1
}

enum F {
    L1 = 1,
    H1 = 0xFFFFFFFFFFFFFFFF
}

static C0: f32 = E::L0 as f32; //~ ERROR casting
static C1: f32 = F::H1 as f32; //~ ERROR casting

pub fn main() {
    let b = C0;
    let d = C1;
    assert_eq!(b, -1.0f32);
    assert_eq!(d, -1.0f32);
}
