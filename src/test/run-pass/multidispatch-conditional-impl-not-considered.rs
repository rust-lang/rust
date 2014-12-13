// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that we correctly ignore the blanket impl
// because (in this case) `T` does not impl `Clone`.
//
// Issue #17594.

use std::cell::RefCell;

trait Foo {
    fn foo(&self) {}
}

impl<T> Foo for T where T: Clone {}

struct Bar;

impl Bar {
    fn foo(&self) {}
}

fn main() {
    let b = RefCell::new(Bar);
    b.borrow().foo();
}
