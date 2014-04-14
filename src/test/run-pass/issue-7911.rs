// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-pretty

// (Closes #7911) Test that we can use the same self expression
// with different mutability in macro in two methods

#![allow(unused_variable)] // unused foobar_immut + foobar_mut
#![feature(macro_rules)]

trait FooBar {}
struct Bar(i32);
struct Foo { bar: Bar }

impl FooBar for Bar {}

trait Test {
    fn get_immut<'r>(&'r self) -> &'r FooBar;
    fn get_mut<'r>(&'r mut self) -> &'r mut FooBar;
}

macro_rules! generate_test(($type_:path, $field:expr) => (
    impl Test for $type_ {
        fn get_immut<'r>(&'r self) -> &'r FooBar {
            &$field as &FooBar
        }

        fn get_mut<'r>(&'r mut self) -> &'r mut FooBar {
            &mut $field as &mut FooBar
        }
    }
))

generate_test!(Foo, self.bar)

pub fn main() {
    let mut foo: Foo = Foo { bar: Bar(42) };
    { let foobar_immut = foo.get_immut(); }
    { let foobar_mut = foo.get_mut(); }
}
