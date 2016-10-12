// Copyright 2013-2015 The Rust Project Developers. See the COPYRIGHT
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

#![feature(plugin, custom_attribute)]
#![plugin(macro_crate_test)]

#[macro_use] #[no_link]
extern crate macro_crate_test;

#[derive(PartialEq, Clone, Debug)]
#[into_multi_foo]
fn foo() -> AnotherFakeTypeThatHadBetterGoAway {}

// Check that the `#[into_multi_foo]`-generated `foo2` is configured away
fn foo2() {}

trait Qux {
    #[into_multi_foo]
    fn bar();
}

impl Qux for i32 {
    #[into_multi_foo]
    fn bar() {}
}

impl Qux for u8 {}

pub fn main() {
    assert_eq!(1, make_a_1!());
    assert_eq!(2, exported_macro!());

    assert_eq!(Foo2::Bar2, Foo2::Bar2);
    test(None::<Foo2>);

    let _ = Foo3::Bar;

    let x = 10i32;
    assert_eq!(x.foo(), 42);
    let x = 10u8;
    assert_eq!(x.foo(), 0);
}

fn test<T: PartialEq+Clone>(_: Option<T>) {}
