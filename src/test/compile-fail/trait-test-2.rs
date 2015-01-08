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
impl bar for isize { fn dup(&self) -> isize { *self } fn blah<X>(&self) {} }
impl bar for usize { fn dup(&self) -> usize { *self } fn blah<X>(&self) {} }

fn main() {
    10is.dup::<isize>(); //~ ERROR does not take type parameters
    10is.blah::<isize, isize>(); //~ ERROR incorrect number of type parameters
    (box 10is as Box<bar>).dup(); //~ ERROR cannot convert to a trait object
}
