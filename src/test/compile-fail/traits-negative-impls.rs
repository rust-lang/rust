// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(optin_builtin_traits)]

use std::marker::Send;

struct Outer<T: Send>(T);

struct TestType;
impl !Send for TestType {}

struct Outer2<T>(T);

unsafe impl<T: Send> Sync for Outer2<T> {}

fn is_send<T: Send>(_: T) {}
fn is_sync<T: Sync>(_: T) {}

fn main() {
    Outer(TestType);
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `TestType`

    is_send(TestType);
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `TestType`

    // This will complain about a missing Send impl because `Sync` is implement *just*
    // for T that are `Send`. Look at #20366 and #19950
    is_sync(Outer2(TestType));
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `TestType`
}
