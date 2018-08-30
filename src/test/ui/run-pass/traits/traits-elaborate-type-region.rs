// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![allow(dead_code)]

// Test that we elaborate `Type: 'region` constraints and infer various important things.

trait Master<'a, T: ?Sized> {
    fn foo() where T: 'a;
}

// [U]: 'a => U: 'a
impl<'a, U> Master<'a, [U]> for () {
    fn foo() where U: 'a { }
}

// &'b U: 'a => 'b: 'a, U: 'a
impl<'a, 'b, U> Master<'a, &'b U> for () {
    fn foo() where 'b: 'a, U: 'a { }
}

// &'b [U]: 'a => 'b: 'a, U: 'a
impl<'a, 'b, U> Master<'a, &'b [U]> for () {
    fn foo() where 'b: 'a, U: 'a { }
}

// Foo<'b>: 'a => 'b: 'a
struct Foo<'a> { x: &'a () }
impl<'a, 'b> Master<'a, Foo<'b>> for () {
    fn foo() where 'b: 'a { }
}

// Bar<'b, T>: 'a => 'b: 'a, T: 'a
struct Bar<'a, T: 'a> { x: &'a T }
impl<'a, 'b, T> Master<'a, Bar<'b, T>> for () {
    fn foo() where 'b: 'a, T: 'a { }
}

// fn(T): 'a => T: 'a
impl<'a, T> Master<'a, fn(T)> for () {
    fn foo() where T: 'a { }
}

// fn() -> T: 'a => T: 'a
impl<'a, T> Master<'a, fn() -> T> for () {
    fn foo() where T: 'a { }
}

fn main() {
    println!("Hello, world!");
}
