// Copyright 2015 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Prefix in imports with empty braces should be resolved and checked privacy, stability, etc.

// aux-build:lint_stability.rs

extern crate lint_stability;

use lint_stability::UnstableStruct::{}; //~ ERROR use of unstable library feature 'test_feature'
use lint_stability::StableStruct::{}; // OK

fn main() {}
