// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// compile-flags: -Z parse-only

fn foo() { //~ NOTE un-closed delimiter
  match Some(x) {
  //~^ NOTE this delimiter might not be properly closed...
      Some(y) => { panic!(); }
      None => { panic!(); }
}
//~^ NOTE ...as it matches this but it has different indentation

fn bar() {
    let mut i = 0;
    while (i < 1000) {}
}

fn main() {} //~ ERROR this file contains an un-closed delimiter
