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

struct Sweet<X>(X);
pub trait Sugar {}
pub trait Fruit {}
impl<T:Sugar> Sweet<T> { fn dummy(&self) { } }
//~^ ERROR E0592
impl<T:Fruit> Sweet<T> { fn dummy(&self) { } }

trait Bar<X> {}
struct A<T, X>(T, X);
impl<X, T> A<T, X> where T: Bar<X> { fn f(&self) {} }
//~^ ERROR E0592
impl<X> A<i32, X> { fn f(&self) {} }

fn main() {}
