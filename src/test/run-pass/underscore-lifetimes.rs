// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(underscore_lifetimes)]

struct Foo<'a>(&'a u8);

fn foo(x: &u8) -> Foo<'_> {
    Foo(x)
}

fn foo2(x: &'_ u8) -> Foo<'_> {
    Foo(x)
}

fn foo3(x: &'_ u8) -> Foo {
    Foo(x)
}

fn foo4(_: Foo<'_>) {}

struct Foo2<'a, 'b> {
    a: &'a u8,
    b: &'b u8,
}
fn foo5<'b>(foo: Foo2<'_, 'b>) -> &'b u8 {
    foo.b
}

fn main() {
    let x = &5;
    let _ = foo(x);
    let _ = foo2(x);
    let _ = foo3(x);
    foo4(Foo(x));
    let _ = foo5(Foo2 {
        a: x,
        b: &6,
    });
}
