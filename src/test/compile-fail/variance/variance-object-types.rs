// Copyright 2012 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that Cell is considered invariant with respect to its
// type.

#![feature(rustc_attrs)]

use std::cell::Cell;

// For better or worse, associated types are invariant, and hence we
// get an invariant result for `'a`.
#[rustc_variance]
struct Foo<'a> { //~ ERROR [o]
    x: Box<Fn(i32) -> &'a i32 + 'static>
}

fn main() {
}
