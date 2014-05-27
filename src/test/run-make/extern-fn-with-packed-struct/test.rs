// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[repr(packed)]
#[deriving(PartialEq, Show)]
struct Foo {
    a: i8,
    b: i16,
    c: i8
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
