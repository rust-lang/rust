// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

struct MyType;

struct MyType1<T>(T);

trait Bar {
    type Out;
}

impl<T> MyType {
    //~^ ERROR  the type parameter `T` is not constrained
}

impl<T> MyType1<T> {
    // OK, T is used in `Foo<T>`.
}

impl<T,U> MyType1<T> {
    //~^ ERROR  the type parameter `U` is not constrained
}

impl<T,U> MyType1<T> where T: Bar<Out=U> {
    // OK, T is used in `Foo<T>`.
}

fn main() { }
