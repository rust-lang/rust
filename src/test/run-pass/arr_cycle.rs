// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::cell::Cell;

#[derive(Debug)]
struct B<'a> {
    a: [Cell<Option<&'a B<'a>>>; 2]
}

impl<'a> B<'a> {
    fn new() -> B<'a> {
        B { a: [Cell::new(None), Cell::new(None)] }
    }
}

fn f() {
    let (b1, b2, b3);
    b1 = B::new();
    b2 = B::new();
    b3 = B::new();
    b1.a[0].set(Some(&b2));
    b1.a[1].set(Some(&b3));
    b2.a[0].set(Some(&b2));
    b2.a[1].set(Some(&b3));
    b3.a[0].set(Some(&b1));
    b3.a[1].set(Some(&b2));
}

fn main() {
    f();
}
