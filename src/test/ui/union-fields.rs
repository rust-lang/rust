// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![deny(dead_code)]

union U {
    x: u32,
    y: f32,
}

struct V {
    x: u32,
    y: u32,
}

fn main() {
    let u = U { x: 0x3f800000 };
    let _f = unsafe { u.y };
    let v = V { x: 0, y: 0 };
    println!("{}", v.x);
}

