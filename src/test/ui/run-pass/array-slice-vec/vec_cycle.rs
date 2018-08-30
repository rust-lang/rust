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
struct C<'a> {
    v: Vec<Cell<Option<&'a C<'a>>>>,
}

impl<'a> C<'a> {
    fn new() -> C<'a> {
        C { v: Vec::new() }
    }
}

fn f() {
    let (mut c1, mut c2, mut c3);
    c1 = C::new();
    c2 = C::new();
    c3 = C::new();

    c1.v.push(Cell::new(None));
    c1.v.push(Cell::new(None));
    c2.v.push(Cell::new(None));
    c2.v.push(Cell::new(None));
    c3.v.push(Cell::new(None));
    c3.v.push(Cell::new(None));

    c1.v[0].set(Some(&c2));
    c1.v[1].set(Some(&c3));
    c2.v[0].set(Some(&c2));
    c2.v[1].set(Some(&c3));
    c3.v[0].set(Some(&c1));
    c3.v[1].set(Some(&c2));
}

fn main() {
    f();
}
