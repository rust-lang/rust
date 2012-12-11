// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// xfail-test
/*
  Ideally, the error about the missing close brace in foo would be reported
  near the corresponding open brace. But currently it's reported at the end.
  xfailed for now (see Issue #2354)
 */
fn foo() { //~ ERROR this open brace is not closed
  match some(x) {
      some(y) { fail; }
      none    { fail; }
}

fn bar() {
    let mut i = 0;
    while (i < 1000) {}
}

fn main() {}
