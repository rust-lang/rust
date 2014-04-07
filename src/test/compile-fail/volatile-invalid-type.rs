// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.
#![allow(experimental)]

//! We need to verify that Volatile only accepts a limited number of types.
//! Thus, user define-able types are not allowed.

use std::unstable::volatile::Volatile;

struct Foo { x: int }

fn main() {
    let foo = Foo { x: 10 };
    let vol = Volatile::new(foo);
    //~^ ERROR failed to find an implementation of trait std::unstable::volatile::VolatileSafe
}
