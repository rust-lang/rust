// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// run-pass
// edition:2018

// Tests that `core` and `std` are always available.
use core::iter;
use std::io;
// FIXME(eddyb) Add a `meta` crate to the distribution.
// use meta;

fn main() {
    for _ in iter::once(()) {
        io::stdout();
    }
}
