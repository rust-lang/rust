// Copyright 2014-2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[warn(clippy::cast_lossless)]
#[allow(clippy::no_effect, clippy::unnecessary_operation)]
fn main() {
    // Test clippy::cast_lossless with casts to floating-point types
    1i8 as f32;
    1i8 as f64;
    1u8 as f32;
    1u8 as f64;
    1i16 as f32;
    1i16 as f64;
    1u16 as f32;
    1u16 as f64;
    1i32 as f64;
    1u32 as f64;
}
