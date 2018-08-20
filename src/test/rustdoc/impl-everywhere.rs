// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![crate_name = "foo"]

pub trait Foo {}
pub trait Foo2 {}

pub struct Bar;

impl Foo for Bar {}
impl Foo2 for Bar {}

// @!has foo/fn.foo.html '//section[@id="main"]//pre' "x: &\'x impl Foo"
// @!has foo/fn.foo.html '//section[@id="main"]//pre' "-> &\'x impl Foo {"
pub fn foo<'x>(x: &'x impl Foo) -> &'x impl Foo {
    x
}

// @!has foo/fn.foo2.html '//section[@id="main"]//pre' "x: &\'x impl Foo"
// @!has foo/fn.foo2.html '//section[@id="main"]//pre' '-> impl Foo2 {'
pub fn foo2<'x>(_x: &'x impl Foo) -> impl Foo2 {
    Bar
}

// @!has foo/fn.foo_foo.html '//section[@id="main"]//pre' '-> impl Foo + Foo2 {'
pub fn foo_foo() -> impl Foo + Foo2 {
    Bar
}

// @!has foo/fn.foo2.html '//section[@id="main"]//pre' "x: &'x (impl Foo + Foo2)"
pub fn foo_foo_foo<'x>(_x: &'x (impl Foo + Foo2)) {
}
