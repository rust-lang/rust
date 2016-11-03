// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#[derive(Debug, PartialEq)]
enum Bar {
    A(i64),
    B(i32),
    C,
}

#[derive(Debug, PartialEq)]
struct Foo(Bar, u8);

static FOO: [Foo; 2] = [Foo(Bar::C, 0), Foo(Bar::C, 0xFF)];

fn main() {
    assert_eq!(&FOO[1],  &Foo(Bar::C, 0xFF));
}
