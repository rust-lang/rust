// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// pretty-expanded FIXME #23616

use std::cell::Cell;

struct C<'a> {
    p: Cell<Option<&'a C<'a>>>,
}

impl<'a> C<'a> {
    fn new() -> C<'a> { C { p: Cell::new(None) } }
}

fn f1() {
    let (c1, c2) = (C::new(), C::new());
    c1.p.set(Some(&c2));
    c2.p.set(Some(&c1));
}

fn f2() {
    let (c1, c2);
    c1 = C::new();
    c2 = C::new();
    c1.p.set(Some(&c2));
    c2.p.set(Some(&c1));
}

fn main() {
    f1();
    f2();
}
