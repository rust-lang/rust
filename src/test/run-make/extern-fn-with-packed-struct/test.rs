// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

use std::fmt;

#[repr(packed)]
#[derive(Copy, Clone)]
struct Foo {
    a: i8,
    b: i16,
    c: i8
}

impl PartialEq for Foo {
    fn eq(&self, other: &Foo) -> bool {
        self.a == other.a && self.b == other.b && self.c == other.c
    }
}

impl fmt::Debug for Foo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let a = self.a;
        let b = self.b;
        let c = self.c;

        f.debug_struct("Foo")
            .field("a", &a)
            .field("b", &b)
            .field("c", &c)
            .finish()
    }
}

#[link(name = "test", kind = "static")]
extern {
    fn foo(f: Foo) -> Foo;
}

fn main() {
    unsafe {
        let a = Foo { a: 1, b: 2, c: 3 };
        let b = foo(a);
        assert_eq!(a, b);
    }
}
