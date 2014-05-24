// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(experimental)]

use std::simd::{i32x4, f32x4, u32x4};

fn eq_u32x4(u32x4(x0, x1, x2, x3): u32x4, u32x4(y0, y1, y2, y3): u32x4) -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
}

fn eq_f32x4(f32x4(x0, x1, x2, x3): f32x4, f32x4(y0, y1, y2, y3): f32x4) -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
}

fn eq_i32x4(i32x4(x0, x1, x2, x3): i32x4, i32x4(y0, y1, y2, y3): i32x4) -> bool {
    (x0 == y0) && (x1 == y1) && (x2 == y2) && (x3 == y3)
}

pub fn main() {
    // arithmetic operators

    assert!(eq_u32x4(u32x4(1, 2, 3, 4) + u32x4(4, 3, 2, 1), u32x4(5, 5, 5, 5)));
    assert!(eq_u32x4(u32x4(4, 5, 6, 7) - u32x4(4, 3, 2, 1), u32x4(0, 2, 4, 6)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) * u32x4(4, 3, 2, 1), u32x4(4, 6, 6, 4)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) & u32x4(4, 3, 2, 1), u32x4(0, 2, 2, 0)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) | u32x4(4, 3, 2, 1), u32x4(5, 3, 3, 5)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) ^ u32x4(4, 3, 2, 1), u32x4(5, 1, 1, 5)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) << u32x4(4, 3, 2, 1), u32x4(16, 16, 12, 8)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) >> u32x4(4, 3, 2, 1), u32x4(0, 0, 0, 2)));

    assert!(eq_i32x4(i32x4(1, 2, 3, 4) + i32x4(4, 3, 2, 1), i32x4(5, 5, 5, 5)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) - i32x4(4, 3, 2, 1), i32x4(-3, -1, 1, 3)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) * i32x4(4, 3, 2, 1), i32x4(4, 6, 6, 4)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) & i32x4(4, 3, 2, 1), i32x4(0, 2, 2, 0)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) | i32x4(4, 3, 2, 1), i32x4(5, 3, 3, 5)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) ^ i32x4(4, 3, 2, 1), i32x4(5, 1, 1, 5)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) << i32x4(4, 3, 2, 1), i32x4(16, 16, 12, 8)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) >> i32x4(4, 3, 2, 1), i32x4(0, 0, 0, 2)));

    assert!(eq_f32x4(f32x4(1.0, 2.0, 3.0, 4.0) + f32x4(4.0, 3.0, 2.0, 1.0),
            f32x4(5.0, 5.0, 5.0, 5.0)));
    assert!(eq_f32x4(f32x4(1.0, 2.0, 3.0, 4.0) - f32x4(4.0, 3.0, 2.0, 1.0),
            f32x4(-3.0, -1.0, 1.0, 3.0)));
    assert!(eq_f32x4(f32x4(1.0, 2.0, 3.0, 4.0) * f32x4(4.0, 3.0, 2.0, 1.0),
            f32x4(4.0, 6.0, 6.0, 4.0)));
    assert!(eq_f32x4(f32x4(1.0, 2.0, 3.0, 4.0) / f32x4(4.0, 4.0, 2.0, 1.0),
            f32x4(0.25, 0.5, 1.5, 4.0)));

    // comparison operators

    assert!(eq_u32x4(u32x4(1, 2, 3, 4) == u32x4(3, 2, 1, 0), u32x4(0, !0, 0, 0)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) != u32x4(3, 2, 1, 0), u32x4(!0, 0, !0, !0)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) < u32x4(3, 2, 1, 0), u32x4(!0, 0, 0, 0)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) <= u32x4(3, 2, 1, 0), u32x4(!0, !0, 0, 0)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) >= u32x4(3, 2, 1, 0), u32x4(0, !0, !0, !0)));
    assert!(eq_u32x4(u32x4(1, 2, 3, 4) > u32x4(3, 2, 1, 0), u32x4(0, 0, !0, !0)));

    assert!(eq_i32x4(i32x4(1, 2, 3, 4) == i32x4(3, 2, 1, 0), i32x4(0, !0, 0, 0)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) != i32x4(3, 2, 1, 0), i32x4(!0, 0, !0, !0)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) < i32x4(3, 2, 1, 0), i32x4(!0, 0, 0, 0)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) <= i32x4(3, 2, 1, 0), i32x4(!0, !0, 0, 0)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) >= i32x4(3, 2, 1, 0), i32x4(0, !0, !0, !0)));
    assert!(eq_i32x4(i32x4(1, 2, 3, 4) > i32x4(3, 2, 1, 0), i32x4(0, 0, !0, !0)));
}
