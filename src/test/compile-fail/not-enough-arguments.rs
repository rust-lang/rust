// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that the only error msg we report is the
// mismatch between the # of params, and not other
// unrelated errors.

fn foo(a: int, b: int, c: int, d:int) {
  fail2!();
}

fn main() {
  foo(1, 2, 3);
  //~^ ERROR this function takes 4 parameters but 3
}
