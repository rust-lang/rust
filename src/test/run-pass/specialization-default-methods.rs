// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// First, test only use of explicit `default` items:

trait Foo {
    fn foo(&self) -> bool;
}

impl<T> Foo for T {
    default fn foo(&self) -> bool { false }
}

impl Foo for i32 {}

impl Foo for i64 {
    fn foo(&self) -> bool { true }
}

fn test_foo() {
    assert!(0i8.foo() == false);
    assert!(0i32.foo() == false);
    assert!(0i64.foo() == true);
}

// Next, test mixture of explicit `default` and provided methods:

trait Bar {
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

fn test_bar() {
    assert!(0u8.bar() == 0);
    assert!(0i32.bar() == 1);
    assert!("hello".bar() == 0);
    assert!(vec![()].bar() == 2);
    assert!(vec![0i32].bar() == 2);
    assert!(vec![0i64].bar() == 3);
}

fn main() {
    test_foo();
    test_bar();
}
