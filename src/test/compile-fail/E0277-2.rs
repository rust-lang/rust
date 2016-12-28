// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct Foo {
    bar: Bar
}

struct Bar {
    baz: Baz
}

struct Baz {
    x: *const u8
}

fn is_send<T: Send>() { }

fn main() {
    is_send::<Foo>();
    //~^ ERROR the trait bound `*const u8: std::marker::Send` is not satisfied in `Foo`
    //~| NOTE within `Foo`, the trait `std::marker::Send` is not implemented for `*const u8`
    //~| NOTE: `*const u8` cannot be sent between threads safely
    //~| NOTE: required because it appears within the type `Baz`
    //~| NOTE: required because it appears within the type `Bar`
    //~| NOTE: required because it appears within the type `Foo`
    //~| NOTE: required by `is_send`
}
