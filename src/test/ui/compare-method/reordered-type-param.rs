// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that ty params get matched correctly when comparing
// an impl against a trait
//
// cc #26111

trait A {
  fn b<C:Clone,D>(&self, x: C) -> C;
}

struct E {
 f: isize
}

impl A for E {
  // n.b. The error message is awful -- see #3404
  fn b<F:Clone,G>(&self, _x: G) -> G { panic!() } //~ ERROR method `b` has an incompatible type
}

fn main() {}
