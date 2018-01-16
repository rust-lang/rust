// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
//

#![warn(overflowing_literals)]
#![warn(const_err)]
#![feature(rustc_attrs)]

#[allow(unused_variables)]
#[rustc_error]
fn main() { //~ ERROR: compilation successful
    let x2: i8 = --128; //~ warn: literal out of range for i8

    let x = -3.40282357e+38_f32; //~ warn: literal out of range for f32
    let x =  3.40282357e+38_f32; //~ warn: literal out of range for f32
    let x = -1.7976931348623159e+308_f64; //~ warn: literal out of range for f64
    let x =  1.7976931348623159e+308_f64; //~ warn: literal out of range for f64
}
