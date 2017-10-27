// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(never_type)]

// Regression test for inhabitedness check. The old
// cache used to cause us to incorrectly decide
// that `test_b` was invalid.

struct Foo {
    field1: !,
    field2: Option<&'static Bar>,
}

struct Bar {
    field1: &'static Foo
}

fn test_a() {
    let x: Option<Foo> = None;
    match x { None => () }
}

fn test_b() {
    let x: Option<Bar> = None;
    match x { None => () }
}

fn main() { }
