// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(core_float)]
#![feature(float_extras)]
#![feature(untagged_unions)]

extern crate core;
use core::num::Float;

union U {
    a: (u8, u8),
    b: u16,
}

union W {
    a: u32,
    b: f32,
}

fn main() {
    unsafe {
        let mut u = U { a: (1, 1) };
        assert_eq!(u.b, (1 << 8) + 1);
        u.b = (2 << 8) + 2;
        assert_eq!(u.a, (2, 2));

        let mut w = W { a: 0b0_11111111_00000000000000000000000 };
        assert_eq!(w.b, f32::infinity());
        w.b = f32::neg_infinity();
        assert_eq!(w.a, 0b1_11111111_00000000000000000000000);
    }
}
