// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #25954: detect and reject a closure type that
// references itself.

use std::cell::{Cell, RefCell};

struct A<T: Fn()> {
    x: RefCell<Option<T>>,
    b: Cell<i32>,
}

fn main() {
    let mut p = A{x: RefCell::new(None), b: Cell::new(4i32)};

    // This is an error about types of infinite size:
    let q = || p.b.set(5i32); //~ ERROR mismatched types

    *(p.x.borrow_mut()) = Some(q);
}
