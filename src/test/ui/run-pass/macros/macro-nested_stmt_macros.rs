// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

macro_rules! foo {
    () => {
        struct Bar;
        struct Baz;
    }
}

macro_rules! grault {
    () => {
        foo!();
        struct Xyzzy;
    }
}

fn static_assert_exists<T>() { }

fn main() {
    grault!();
    static_assert_exists::<Bar>();
    static_assert_exists::<Baz>();
    static_assert_exists::<Xyzzy>();
}
