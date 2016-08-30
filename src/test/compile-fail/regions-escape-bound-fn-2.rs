// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

fn with_int<F>(f: F) where F: FnOnce(&isize) {
    let x = 3;
    f(&x);
}

fn main() {
    let mut x = None;
    with_int(|y| x = Some(y));
         //~^ ERROR cannot infer an appropriate lifetime due to conflicting requirements
         //~| ERROR cannot infer an appropriate lifetime due to conflicting requirements
         //~| NOTE cannot infer an appropriate lifetime
         //~| NOTE first, the lifetime cannot outlive the anonymous lifetime #1
         //~| NOTE ...so that expression is assignable (expected &isize, found &isize)
         //~| NOTE but, the lifetime must be valid for the expression at
         //~| NOTE ...so that a type/lifetime parameter is in scope here
}
