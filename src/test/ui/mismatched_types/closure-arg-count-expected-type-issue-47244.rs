// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Regression test for #47244: in this specific scenario, when the
// expected type indicated 1 argument but the closure takes two, we
// would (early on) create type variables for the type of `b`. If the
// user then attempts to invoke a method on `b`, we would get an error
// saying that the type of `b` must be known, which was not very
// helpful.

// run-rustfix

use std::collections::HashMap;

fn main() {
    let mut m = HashMap::new();
    m.insert("foo", "bar");

    let _n = m.iter().map(|_, b| {
        //~^ ERROR closure is expected to take a single 2-tuple
        b.to_string()
    });
}
