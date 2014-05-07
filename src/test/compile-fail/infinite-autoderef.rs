// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// error-pattern: reached the recursion limit while auto-dereferencing

use std::ops::Deref;

struct Foo;

impl Deref<Foo> for Foo {
    fn deref<'a>(&'a self) -> &'a Foo {
        self
    }
}

pub fn main() {
    let mut x;
    loop {
        x = box x;
        x.foo;
        x.bar();
    }

    Foo.foo;
    Foo.bar();
}
