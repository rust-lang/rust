// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

trait A {
    type Output;
    fn a(&self) -> <Self as A>::X;
    //~^ ERROR cannot find associated type `X` in trait `A`
}

impl A for u32 {
    type Output = u32;
    fn a(&self) -> u32 {
        0
    }
}

fn main() {
    let a: u32 = 0;
    let b: u32 = a.a();
}
