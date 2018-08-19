// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Tests that we consider `T: Sugar + Fruit` to be ambiguous, even
// though no impls are found.

pub trait Sugar {}
pub trait Fruit {}
pub trait Sweet {}
impl<T:Sugar> Sweet for T { }
impl<T:Fruit> Sweet for T { }
//~^ ERROR E0119

pub trait Foo<X> {}
pub trait Bar<X> {}
impl<X, T> Foo<X> for T where T: Bar<X> {}
impl<X> Foo<X> for i32 {}
//~^ ERROR E0119

fn main() { }
