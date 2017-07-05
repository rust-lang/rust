// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that shadowed lifetimes generate an error.

#![feature(box_syntax)]

struct Foo<T>(T);

impl<T> Foo<T> {
    fn shadow_in_method<T>(&self) {}
    //~^ ERROR type parameter `T` shadows another type parameter

    fn not_shadow_in_item<U>(&self) {
        struct Bar<T, U>(T,U); // not a shadow, separate item
        fn foo<T, U>() {} // same
    }
}

trait Bar<T> {
    fn dummy(&self) -> T;

    fn shadow_in_required<T>(&self);
    //~^ ERROR type parameter `T` shadows another type parameter

    fn shadow_in_provided<T>(&self) {}
    //~^ ERROR type parameter `T` shadows another type parameter

    fn not_shadow_in_required<U>(&self);
    fn not_shadow_in_provided<U>(&self) {}
}

fn main() {}
