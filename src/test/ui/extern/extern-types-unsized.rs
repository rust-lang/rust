// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Make sure extern types are !Sized.

#![feature(extern_types)]

extern {
    type A;
}

struct Foo {
    x: u8,
    tail: A,
}

struct Bar<T: ?Sized> {
    x: u8,
    tail: T,
}

fn assert_sized<T>() { }

fn main() {
    assert_sized::<A>();
    //~^ ERROR the size for values of type

    assert_sized::<Foo>();
    //~^ ERROR the size for values of type

    assert_sized::<Bar<A>>();
    //~^ ERROR the size for values of type

    assert_sized::<Bar<Bar<A>>>();
    //~^ ERROR the size for values of type
}
