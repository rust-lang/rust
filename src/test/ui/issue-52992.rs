// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for an NLL-related ICE (#52992) -- computing
// implied bounds was causing outlives relations that were not
// properly handled.
//
// compile-pass

#![feature(nll)]

fn main() {}

fn fail<'a>() -> Struct<'a, Generic<()>> {
    Struct(&Generic(()))
}

struct Struct<'a, T>(&'a T) where
    T: Trait + 'a,
    T::AT: 'a; // only fails with this bound

struct Generic<T>(T);

trait Trait {
    type AT;
}

impl<T> Trait for Generic<T> {
    type AT = T; // only fails with a generic AT
}
