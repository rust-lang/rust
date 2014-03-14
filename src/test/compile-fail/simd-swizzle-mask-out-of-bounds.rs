// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[feature(simd, phase)];

#[phase(syntax)]
extern crate simd_syntax;
extern crate simd;

fn main() {
    let v = gather_simd!(0.0f32, 0.0f32, 0.0f32, 0.0f32);
    let _ = swizzle_simd!(v -> (15));
    //~^ ERROR invalid mask index in SIMD swizzle: `15`
    let _ = swizzle_simd!(v -> (0, 0, 0, 0, 10));
    //~^ ERROR invalid mask index in SIMD swizzle: `10`
}
