// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(unboxed_closures)]

use std::ops::FnMut;

 fn make_adder(x: int) -> Box<FnMut<(int,),int>+'static> {
    (box move |&mut: y: int| -> int { x + y }) as
        Box<FnMut<(int,),int>+'static>
}

pub fn main() {
    let mut adder = make_adder(3);
    let z = adder.call_mut((2,));
    println!("{}", z);
    assert_eq!(z, 5);
}

