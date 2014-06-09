// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-stage1
// ignore-pretty

#![feature(phase)]
#[phase(plugin)]
extern crate hexfloat;

pub fn main() {
    let a = hexfloat!("0x1.999999999999ap-4");
    assert_eq!(a, 0.1);
    let b = hexfloat!("-0x1.fffp-4", f32);
    assert_eq!(b, -0.12498474_f32);
    let c = hexfloat!("0x.12345p5", f64);
    let d = hexfloat!("0x0.12345p5", f64);
    assert_eq!(c,d);
    let f = hexfloat!("0x10.p4", f32);
    let g = hexfloat!("0x10.0p4", f32);
    assert_eq!(f,g);
}
