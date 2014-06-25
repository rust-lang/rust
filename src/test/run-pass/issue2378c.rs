// Copyright 2012-2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue2378a.rs
// aux-build:issue2378b.rs

extern crate issue2378a;
extern crate issue2378b;

use issue2378a::{just};
use issue2378b::{two_maybes};

pub fn main() {
    let x = two_maybes{a: just(3i), b: just(5i)};
    assert_eq!(x[0u], (3, 5));
}
