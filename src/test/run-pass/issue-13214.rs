// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// defining static with struct that contains enum
// with &'static str variant used to cause ICE

pub enum Foo {
    Bar,
    Baz(&'static str),
}

pub static TEST: Test = Test {
    foo: Bar,
    c: 'a'
};

pub struct Test {
    foo: Foo,
    c: char,
}

fn main() {}
