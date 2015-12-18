// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Check that we require that associated types in an impl are well-formed.

#![feature(rustc_attrs)]
#![allow(dead_code)]

pub trait MyHash { }

pub struct MySet<T:MyHash> {
    data: Vec<T>
}

pub trait Foo {
    type Bar;
}

impl<T> Foo for T {
    type Bar = MySet<T>;
    //~^ ERROR the trait `MyHash` is not implemented for the type `T`
}

#[rustc_error]
fn main() { }

