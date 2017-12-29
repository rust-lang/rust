// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]
use std::any::Any;

fn main()
{
    fn h(x:i32) -> i32 {3*x}
    let mut vfnfer:Vec<Box<Any>> = vec![];
    vfnfer.push(box h);
    println!("{:?}",(vfnfer[0] as Fn)(3));
    //~^ ERROR the precise format of `Fn`-family traits'
    //~| ERROR wrong number of type arguments: expected 1, found 0 [E0243]
    //~| ERROR the value of the associated type `Output` (from the trait `std::ops::FnOnce`)
}
