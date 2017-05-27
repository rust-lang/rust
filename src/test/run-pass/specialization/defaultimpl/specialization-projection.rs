// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(specialization)]

// Make sure we *can* project non-defaulted associated types
// cf compile-fail/specialization-default-projection.rs

// First, do so without any use of specialization

trait Foo {
    type Assoc;
}

impl<T> Foo for T {
    type Assoc = ();
}

fn generic_foo<T>() -> <T as Foo>::Assoc {
    ()
}

// Next, allow for one layer of specialization

trait Bar {
    type Assoc;
}

default impl<T> Bar for T {
    type Assoc = ();
}

impl<T: Clone> Bar for T {
    type Assoc = u8;
}

fn generic_bar_clone<T: Clone>() -> <T as Bar>::Assoc {
    0u8
}

fn main() {
}
