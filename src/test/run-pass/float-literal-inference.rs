// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct S {
    z: f64
}

pub fn main() {
    let x: f32 = 4.0;
    println(x.to_str());
    let y: f64 = 64.0;
    println(y.to_str());
    let z = S { z: 1.0 };
    println(z.z.to_str());
}
