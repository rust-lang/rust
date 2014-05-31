// Copyright 2013-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:macro_crate_test.rs
// ignore-stage1

#![feature(phase)]

#[phase(syntax)]
extern crate macro_crate_test;

#[into_foo]
#[deriving(PartialEq, Clone, Show)]
fn foo() -> AFakeTypeThatHadBetterGoAway {}

pub fn main() {
    assert_eq!(1, make_a_1!());
    assert_eq!(2, exported_macro!());

    assert_eq!(Bar, Bar);
    test(None::<Foo>);
}

fn test<T: PartialEq+Clone>(_: Option<T>) {}
