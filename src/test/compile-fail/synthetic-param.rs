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

fn func<#[rustc_synthetic] T>(_: T) {}

struct Foo;

impl Foo {
    pub fn func<#[rustc_synthetic] T>(_: T) {}
}

struct Bar<S> {
    t: S
}

impl<S> Bar<S> {
    pub fn func<#[rustc_synthetic] T>(_: T) {}
}

fn main() {
    func::<u8>(42); //~ ERROR cannot provide explicit type parameters
    func(42); // Ok

    Foo::func::<u8>(42); //~ ERROR cannot provide explicit type parameters
    Foo::func(42); // Ok

    Bar::<i8>::func::<u8>(42); //~ ERROR cannot provide explicit type parameters
    Bar::<i8>::func(42); // Ok
}
