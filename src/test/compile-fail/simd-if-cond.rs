// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// force-host
// xfail-stage1

#[feature(simd, phase)];
#[allow(experimental)];
#[phase(syntax)]
extern crate simd_syntax;
extern crate simd;

#[simd]
#[deriving(Show)]
struct RGBA {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

fn main() {
    // this should be Okay:
    let r = gather_simd!(0.0f32) == gather_simd!(1.0f32);
    let _ = if r[0] { Some(true) }
            else { None };

    if gather_simd!(0.0f32) == gather_simd!(1.0f32) {
        //~^ ERROR expected `bool` but found `<bool, ..1>` (expected bool but found internal simd)
    }

    // simd structures should still require impls of the respective cmps.
    let v1 = RGBA{ r: 0.0f32, g: 0.25f32, b: 0.5f32, a: 0.75f32 };
    let v2 = RGBA{ r: 0.0f32, g: 0.25f32, b: 0.5f32, a: 0.75f32 };
    let _ = v1 == v2;
    //~^ ERROR binary operation `==` cannot be applied to type `RGBA`
}
