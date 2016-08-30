// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

trait Deref {
    fn get(self) -> isize;
}

impl<'a> Deref for &'a isize {
    fn get(self) -> isize {
        *self
    }
}

fn with<R:Deref, F>(f: F) -> isize where F: FnOnce(&isize) -> R {
    f(&3).get()
}

fn return_it() -> isize {
    with(|o| o)
        //~^ ERROR cannot infer an appropriate lifetime due to conflicting requirements
        //~| ERROR cannot infer an appropriate lifetime due to conflicting requirements
        //~| ERROR cannot infer an appropriate lifetime due to conflicting requirements
        //~| NOTE cannot infer an appropriate lifetime
        //~| NOTE first, the lifetime cannot outlive the anonymous lifetime #1 defined on the block
        //~| NOTE ...so that expression is assignable (expected &isize, found &isize)
        //~| NOTE but, the lifetime must be valid for the expression
        //~| NOTE ...so that a type/lifetime parameter is in scope here
}

fn main() {
}
