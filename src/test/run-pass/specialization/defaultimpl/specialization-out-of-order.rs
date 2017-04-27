// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that you can list the more specific impl before the more general one.

#![feature(specialization)]

trait Foo {
    type Out;
}

impl Foo for bool {
    type Out = ();
}

default impl<T> Foo for T {
    type Out = bool;
}

fn main() {}
