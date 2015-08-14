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

struct Outer2<T>(T);

unsafe impl<T: Send> Sync for Outer2<T> {}

fn is_send<T: Send>(_: T) {}
fn is_sync<T: Sync>(_: T) {}

fn dummy() {
    struct TestType;
    impl !Send for TestType {}

    Outer(TestType);
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `dummy::TestType`
    //~| ERROR the trait `core::marker::Send` is not implemented for the type `dummy::TestType`
}

fn dummy1b() {
    struct TestType;
    impl !Send for TestType {}

    is_send(TestType);
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `dummy1b::TestType`
}

fn dummy1c() {
    struct TestType;
    impl !Send for TestType {}

    is_send((8, TestType));
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `dummy1c::TestType`
}

fn dummy2() {
    struct TestType;
    impl !Send for TestType {}

    is_send(Box::new(TestType));
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `dummy2::TestType`
}

fn dummy3() {
    struct TestType;
    impl !Send for TestType {}

    is_send(Box::new(Outer2(TestType)));
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `dummy3::TestType`
}

fn main() {
    struct TestType;
    impl !Send for TestType {}

    // This will complain about a missing Send impl because `Sync` is implement *just*
    // for T that are `Send`. Look at #20366 and #19950
    is_sync(Outer2(TestType));
    //~^ ERROR the trait `core::marker::Send` is not implemented for the type `main::TestType`
}
