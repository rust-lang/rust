// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Creating a stack closure which references an owned pointer and then
// transferring ownership of the owned box before invoking the stack
// closure results in a crash.

fn twice(x: ~uint) -> uint {
     *x * 2
}

fn invoke(f: || -> uint) {
     f();
}

fn main() {
      let x  : ~uint         = ~9;
      let sq : || -> uint =  || { *x * *x };

      twice(x); //~ ERROR: cannot move out of
      invoke(sq);
}
