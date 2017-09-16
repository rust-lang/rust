// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that impls are allowed to have looser, more permissive bounds
// than the traits require.


trait A {
  fn b<C:Sync,D>(&self, x: C) -> C;
}

struct E {
 f: isize
}

impl A for E {
  fn b<F,G>(&self, _x: F) -> F { panic!() }
}

pub fn main() {}
