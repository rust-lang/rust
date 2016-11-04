// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(more_struct_aliases)]

struct Foo<A> { inner: A }

trait Bar { fn bar(); }

impl Bar for Foo<i32> {
    fn bar() {
        Self { inner: 1.5f32 }; //~ ERROR mismatched types
                                //~^ NOTE expected i32, found f32
    }
}

impl<T> Foo<T> {
    fn new<U>(u: U) -> Foo<U> {
        Self {
        //~^ ERROR mismatched types
        //~| expected type parameter, found a different type parameter
        //~| expected type `Foo<U>`
        //~| found type `Foo<T>`
            inner: u
            //~^ ERROR mismatched types
            //~| expected type parameter, found a different type parameter
            //~| expected type `T`
            //~| found type `U`
        }
    }
}

fn main() {}
