// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


// checks that struct simd types are not implicitly indexable.

#[feature(simd)];
#[allow(experimental)];

#[simd]
struct RGBA {
    r: f32,
    g: f32,
    b: f32,
    a: f32,
}

#[inline(never)]
fn index(v: &mut RGBA) {
    v[0] = 10.0f32;
    //~^ ERROR
}


pub fn main() {
    let mut v = RGBA {
        r: 20.0f32,
        g: 20.0f32,
        b: 20.0f32,
        a: 20.0f32,
    };
    index(&mut v);
}
