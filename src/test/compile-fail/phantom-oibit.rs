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
// the `PhantomData<T>` type itself (which almost always implements a "default" trait
// (`impl Trait for ..`))

#![feature(optin_builtin_traits)]

use std::marker::{PhantomData};

unsafe trait Zen {}

unsafe impl Zen for .. {}

unsafe impl<'a, T: 'a> Zen for &'a T where T: Sync {}

struct Guard<'a, T: 'a> {
    _marker: PhantomData<&'a T>,
}

struct Nested<T>(T);

fn is_zen<T: Zen>(_: T) {}

fn not_sync<T>(x: Guard<T>) {
    is_zen(x)  //~ error: the trait `core::marker::Sync` is not implemented for the type `T`
}

fn nested_not_sync<T>(x: Nested<Guard<T>>) {
    is_zen(x)  //~ error: the trait `core::marker::Sync` is not implemented for the type `T`
}

fn main() {}
