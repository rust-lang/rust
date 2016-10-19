// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Test that a local type (with no type parameters) appearing within a
// *non-fundamental* remote type like `Vec` is not considered local.

// aux-build:coherence_lib.rs

extern crate coherence_lib as lib;
use lib::Remote;

struct Local;

impl Remote for Vec<Local> { } //~ ERROR E0117

fn main() { }
