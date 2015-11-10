// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(box_syntax)]

trait bar { fn dup(&self) -> Self; fn blah<X>(&self); }
impl bar for i32 { fn dup(&self) -> i32 { *self } fn blah<X>(&self) {} }
impl bar for u32 { fn dup(&self) -> u32 { *self } fn blah<X>(&self) {} }

fn main() {
    10.dup::<i32>(); //~ ERROR does not take type parameters
    10.blah::<i32, i32>();
    //~^ ERROR incorrect number of type parameters given for this method: expected 1, found 2
    (box 10 as Box<bar>).dup();
    //~^ ERROR E0038
    //~| ERROR E0038
    //~| ERROR E0277
    //~| WARNING E0038
}
