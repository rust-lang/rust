// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

/* Make sure a loop{} with a break in it can't be
   the tailexpr in the body of a diverging function */
fn forever() -> ! {
  loop {
    break;
  }
  return 42i; //~ ERROR expected `!`, found `int`
}

fn main() {
  if (1 == 2) { forever(); }
}
