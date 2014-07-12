// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.



// Regression test for issue #15557

#![allow(dead_code)]
struct AReg1<'a>(&'a u32);

impl<'a> Drop for AReg1<'a> {
//~^ ERROR: cannot implement a destructor on a structure with type parameters
  fn drop(&mut self) {}
}

fn main() {}
