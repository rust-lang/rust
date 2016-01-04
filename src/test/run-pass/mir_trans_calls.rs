// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(rustc_attrs)]

#[rustc_mir]
fn test1(a: isize, b: (i32, i32), c: &[i32]) -> (isize, (i32, i32), &[i32]) {
    // Test passing a number of arguments including a fat pointer.
    // Also returning via an out pointer
    fn callee(a: isize, b: (i32, i32), c: &[i32]) -> (isize, (i32, i32), &[i32]) {
        (a, b, c)
    }
    callee(a, b, c)
}

#[rustc_mir]
fn test2(a: isize) -> isize {
    // Test passing a single argument.
    // Not using out pointer.
    fn callee(a: isize) -> isize {
        a
    }
    callee(a)
}

struct Foo;
impl Foo {
    fn inherent_method(&self, a: isize) -> isize { a }
}

#[rustc_mir]
fn test3(x: &Foo, a: isize) -> isize {
    // Test calling inherent method
    x.inherent_method(a)
}

trait Bar {
    fn extension_method(&self, a: isize) -> isize { a }
}
impl Bar for Foo {}

#[rustc_mir]
fn test4(x: &Foo, a: isize) -> isize {
    // Test calling extension method
    x.extension_method(a)
}

#[rustc_mir]
fn test5(x: &Bar, a: isize) -> isize {
    // Test calling method on trait object
    x.extension_method(a)
}

// FIXME #30661: Although this function has the #[rustc_mir] attribute it never
//               was translated via the MIR implementation because attributes
//               where not passed along to trans::base::trans_fn() for generic
//               functions.
//               Uncomment this test once the thing it tests is fixed.
// #[rustc_mir]
// fn test6<T: Bar>(x: &T, a: isize) -> isize {
//     // Test calling extension method on generic callee
//     x.extension_method(a)
// }

trait One<T = Self> {
    fn one() -> T;
}
impl One for isize {
    fn one() -> isize { 1 }
}

#[rustc_mir]
fn test7() -> isize {
    // Test calling trait static method
    <isize as One>::one()
}

struct Two;
impl Two {
    fn two() -> isize { 2 }
}

#[rustc_mir]
fn test8() -> isize {
    // Test calling impl static method
    Two::two()
}

#[rustc_mir]
fn test_fn_impl(f: &&Fn(i32, i32) -> i32, x: i32, y: i32) -> i32 {
    // This call goes through the Fn implementation for &Fn provided in
    // core::ops::impls. It expands to a static Fn::call() that calls the
    // Fn::call() implemenation of the object shim underneath.
    f(x, y)
}

#[rustc_mir]
fn test_fn_object(f: &Fn(i32, i32) -> i32, x: i32, y: i32) -> i32 {
    f(x, y)
}

fn main() {
    assert_eq!(test1(1, (2, 3), &[4, 5, 6]), (1, (2, 3), &[4, 5, 6][..]));
    assert_eq!(test2(98), 98);
    assert_eq!(test3(&Foo, 42), 42);
    assert_eq!(test4(&Foo, 970), 970);
    assert_eq!(test5(&Foo, 8576), 8576);
    // see definition of test6() above
    // assert_eq!(test6(&Foo, 12367), 12367);
    assert_eq!(test7(), 1);
    assert_eq!(test8(), 2);

    let function_object = (&|x: i32, y: i32| { x + y }) as &Fn(i32, i32) -> i32;
    assert_eq!(test_fn_object(function_object, 100, 1), 101);
    assert_eq!(test_fn_impl(&function_object, 100, 2), 102);
}
