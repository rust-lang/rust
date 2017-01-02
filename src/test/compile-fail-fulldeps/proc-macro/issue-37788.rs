// Copyright 2016 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:derive-a-b.rs

#[macro_use]
extern crate derive_a_b;

fn main() {
    // Test that constructing the `visible_parent_map` (in `cstore_impl.rs`) does not ICE.
    std::cell::Cell::new(0) //~ ERROR mismatched types
}
