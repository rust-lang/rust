// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::cmp_nan)]
#[allow(clippy::float_cmp, clippy::no_effect, clippy::unnecessary_operation)]
fn main() {
    let x = 5f32;
    x == std::f32::NAN;
    x != std::f32::NAN;
    x < std::f32::NAN;
    x > std::f32::NAN;
    x <= std::f32::NAN;
    x >= std::f32::NAN;

    let y = 0f64;
    y == std::f64::NAN;
    y != std::f64::NAN;
    y < std::f64::NAN;
    y > std::f64::NAN;
    y <= std::f64::NAN;
    y >= std::f64::NAN;
}
