// Copyright 2017 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-cloudabi no std::fs support

#![feature(try_trait)]

use std::ops::Try;

fn main() {
    // error for a `Try` type on a non-`Try` fn
    std::fs::File::open("foo")?; //~ ERROR the `?` operator can only

    // a non-`Try` type on a non-`Try` fn
    ()?; //~ ERROR the `?` operator can only

    // an unrelated use of `Try`
    try_trait_generic::<()>(); //~ ERROR the trait bound
}



fn try_trait_generic<T: Try>() -> T {
    // and a non-`Try` object on a `Try` fn.
    ()?; //~ ERROR the `?` operator can only

    loop {}
}
