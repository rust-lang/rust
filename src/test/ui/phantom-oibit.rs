// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Ensure that OIBIT checks `T` when it encounters a `PhantomData<T>` field, instead of checking
// the `PhantomData<T>` type itself (which almost always implements an auto trait)

#![feature(optin_builtin_traits)]

use std::marker::{PhantomData};

unsafe auto trait Zen {}

unsafe impl<'a, T: 'a> Zen for &'a T where T: Sync {}

struct Guard<'a, T: 'a> {
    _marker: PhantomData<&'a T>,
}

struct Nested<T>(T);

fn is_zen<T: Zen>(_: T) {}

fn not_sync<T>(x: Guard<T>) {
    is_zen(x)
    //~^ ERROR `T` cannot be shared between threads safely [E0277]
}

fn nested_not_sync<T>(x: Nested<Guard<T>>) {
    is_zen(x)
    //~^ ERROR `T` cannot be shared between threads safely [E0277]
}

fn main() {}
