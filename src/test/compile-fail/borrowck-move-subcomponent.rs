// Copyright 2012-2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that the borrow checker checks all components of a path when moving
// out.

#[no_std];

struct S {
  x : ~int
}

fn f<T>(_: T) {}

fn main() {
  let a : S = S { x : ~1 };
  let pb = &a;
  let S { x: ax } = a;  //~ ERROR cannot move out
  f(pb);
}
