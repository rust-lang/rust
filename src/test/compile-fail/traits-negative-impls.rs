// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// The dummy functions are used to avoid adding new cfail files.
// What happens is that the compiler attempts to squash duplicates and some
// errors are not reported. This way, we make sure that, for each function, different
// typeck phases are involved and all errors are reported.

#![feature(optin_builtin_traits)]

use std::marker::Send;

struct Outer<T: Send>(T);

struct TestType;
impl !Send for TestType {}

struct Outer2<T>(T);

unsafe impl<T: Send> Sync for Outer2<T> {}

fn is_send<T: Send>(_: T) {}
fn is_sync<T: Sync>(_: T) {}

fn dummy() {
    Outer(TestType);
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `TestType`

    is_send(TestType);
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `TestType`

    is_send((8, TestType));
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `TestType`
}

fn dummy2() {
    is_send(Box::new(TestType));
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `TestType`
}

fn dummy3() {
    is_send(Box::new(Outer2(TestType)));
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `TestType`
}

fn main() {
    // This will complain about a missing Send impl because `Sync` is implement *just*
    // for T that are `Send`. Look at #20366 and #19950
    is_sync(Outer2(TestType));
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `TestType`
}
