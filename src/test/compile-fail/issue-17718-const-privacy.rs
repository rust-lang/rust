// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:issue-17718-const-privacy.rs

extern crate "issue-17718-const-privacy" as other;

use a::B; //~ ERROR: const `B` is private
use other::{
    FOO,
    BAR, //~ ERROR: const `BAR` is private
    FOO2,
};

mod a {
    const B: uint = 3;
}

fn main() {}
