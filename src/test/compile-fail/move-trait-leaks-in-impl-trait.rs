// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

#![feature(conservative_impl_trait, immovable_types)]

use std::marker::{Move, Immovable};

trait Foo: ?Move {}

impl<T: ?Move> Foo for T {}

fn immovable() -> impl Foo {
    Immovable
}

fn movable() -> impl Foo {
    true
}

fn test<T>(_: T) {}

fn main() {
    test(immovable());
    //~^ ERROR the trait bound `std::marker::Immovable: std::marker::Move` is not satisfied
    test(movable());
}
