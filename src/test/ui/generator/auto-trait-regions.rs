// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(generators)]
#![feature(optin_builtin_traits)]

auto trait Foo {}

struct No;

impl !Foo for No {}

struct A<'a, 'b>(&'a mut bool, &'b mut bool, No);

impl<'a, 'b: 'a> Foo for A<'a, 'b> {}

struct OnlyFooIfStaticRef(No);
impl Foo for &'static OnlyFooIfStaticRef {}

struct OnlyFooIfRef(No);
impl<'a> Foo for &'a OnlyFooIfRef {}

fn assert_foo<T: Foo>(f: T) {}

fn main() {
    // Make sure 'static is erased for generator interiors so we can't match it in trait selection
    let x: &'static _ = &OnlyFooIfStaticRef(No);
    let gen = || {
        let x = x;
        yield;
        assert_foo(x);
    };
    assert_foo(gen); //~ ERROR the trait bound `No: Foo` is not satisfied

    // Allow impls which matches any lifetime
    let x = &OnlyFooIfRef(No);
    let gen = || {
        let x = x;
        yield;
        assert_foo(x);
    };
    assert_foo(gen); // ok

    // Disallow impls which relates lifetimes in the generator interior
    let gen = || {
        let a = A(&mut true, &mut true, No);
        yield;
        assert_foo(a);
    };
    assert_foo(gen); //~ ERROR the requirement `for<'r, 's> 'r : 's` is not satisfied
}
