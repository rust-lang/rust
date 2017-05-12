// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-foo.rs
// ignore-stage1
// pp-exact
// Testing that both the inner item and next outer item are
// preserved, and that the first outer item parsed in main is not
// accidentally carried over to each inner function

#[macro_use]
extern crate derive_foo;

#[derive(Foo)]
struct X;

#[derive(Foo)]
#[Bar]
struct Y;

#[derive(Foo)]
struct WithRef {
    x: X,
    #[Bar]
    y: Y,
}

#[derive(Foo)]
enum Enum {

    #[Bar]
    Asdf,
    Qwerty,
}

fn main() { }
