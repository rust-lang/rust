// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:lib.rs

// Regression test for #37291. The problem was that the starting
// environment for a specialization check was not including the
// where-clauses from the impl when attempting to normalize the impl's
// trait-ref, so things like `<C as Foo>::Item` could not resolve,
// since the `C: Foo` trait bound was not included in the environment.

extern crate lib;

use lib::{CV, WrapperB, WrapperC};

fn main() {
    let a = WrapperC(CV);
    let b = WrapperC(CV);
    if false {
        let _ = a * b;
    }
}
