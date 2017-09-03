// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that traits can be implemented for extern types.

#![feature(extern_types)]

extern {
    type A;
}

trait Foo {
    fn foo(&self) { }
}

impl Foo for A {
    fn foo(&self) { }
}

fn assert_foo<T: ?Sized + Foo>() { }

fn use_foo<T: ?Sized + Foo>(x: &Foo) {
    x.foo();
}

fn main() {
    assert_foo::<A>();
}
