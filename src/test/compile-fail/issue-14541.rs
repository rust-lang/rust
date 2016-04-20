// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct vec2 { y: f32 }
struct vec3 { y: f32, z: f32 }

fn make(v: vec2) {
    let vec3 { y: _, z: _ } = v;
    //~^ ERROR mismatched types
    //~| expected type `vec2`
    //~| found type `vec3`
    //~| expected struct `vec2`, found struct `vec3`
}

fn main() { }
