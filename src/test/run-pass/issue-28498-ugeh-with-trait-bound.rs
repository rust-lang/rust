// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Demonstrate the use of the unguarded escape hatch with a trait bound
// to assert that destructor will not access any dead data.
//
// Compare with compile-fail/issue28498-reject-trait-bound.rs

#![feature(dropck_parametricity)]

use std::fmt;

#[derive(Debug)]
struct ScribbleOnDrop(String);

impl Drop for ScribbleOnDrop {
    fn drop(&mut self) {
        self.0 = format!("DROPPED");
    }
}

struct Foo<T:fmt::Debug>(u32, T);

impl<T:fmt::Debug> Drop for Foo<T> {
    #[unsafe_destructor_blind_to_params]
    fn drop(&mut self) {
        // Use of `unsafe_destructor_blind_to_params` is sound,
        // because destructor never accesses the `Debug::fmt` method
        // of `T`, despite having it available.
        println!("Dropping Foo({}, _)", self.0);
    }
}

fn main() {
    let (last_dropped, foo0);
    let (foo1, first_dropped);

    last_dropped = ScribbleOnDrop(format!("last"));
    first_dropped = ScribbleOnDrop(format!("first"));
    foo0 = Foo(0, &last_dropped);
    foo1 = Foo(1, &first_dropped);

    println!("foo0.1: {:?} foo1.1: {:?}", foo0.1, foo1.1);
}
