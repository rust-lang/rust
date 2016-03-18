// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.


#![feature(specialization)]

// First, test only use of explicit `default` items:

pub trait Foo {
    fn foo(&self) -> bool;
}

impl<T> Foo for T {
    default fn foo(&self) -> bool { false }
}

impl Foo for i32 {}

impl Foo for i64 {
    fn foo(&self) -> bool { true }
}

// Next, test mixture of explicit `default` and provided methods:

pub trait Bar {
    fn bar(&self) -> i32 { 0 }
}

impl<T> Bar for T {} // use the provided method

impl Bar for i32 {
    fn bar(&self) -> i32 { 1 }
}
impl<'a> Bar for &'a str {}

impl<T> Bar for Vec<T> {
    default fn bar(&self) -> i32 { 2 }
}
impl Bar for Vec<i32> {}
impl Bar for Vec<i64> {
    fn bar(&self) -> i32 { 3 }
}
