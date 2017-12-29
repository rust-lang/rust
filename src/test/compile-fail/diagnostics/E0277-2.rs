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
}
