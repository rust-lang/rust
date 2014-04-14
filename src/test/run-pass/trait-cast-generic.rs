// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Testing casting of a generic Struct to a Trait with a generic method.
// This is test for issue 10955.
#![allow(unused_variable)]

trait Foo {
    fn f<A>(a: A) -> A {
        a
    }
}

struct Bar<T> {
    x: T,
}

impl<T> Foo for Bar<T> { }

pub fn main() {
    let a = Bar { x: 1 };
    let b = &a as &Foo;
}
