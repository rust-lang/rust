// Copyright 2013 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// aux-build:ifmt_bad_trait1.rs
// aux-build:ifmt_bad_trait2.rs

extern mod ifmt_bad_trait1;
extern mod ifmt_bad_trait2;

use std::fmt::Formatter;

#[fmt="fmt1"]
trait A {} //~ ERROR: no function named `fmt`

#[fmt] //~ ERROR: fmt attribute must have a value specified
trait B {}

#[fmt="a"] //~ ERROR: fmt attribute can only be specified on traits
fn a() {}

#[fmt="fmt2"]
trait C { fn fmt(&Self, &mut Formatter); }
//~^ NOTE: previous definition here
#[fmt="fmt2"]
trait D { fn fmt(&Self, &mut Formatter); }
//~^ ERROR: duplicate fmt trait for `fmt2`

#[fmt="d"]
trait E { fn fmt(&Self, &mut Formatter); }
//~^ ERROR: duplicate fmt trait for `d`

#[fmt="fmt3"]
trait F { unsafe fn fmt(&Self, &mut Formatter); }
//~^ ERROR: the `fmt` function must not be unsafe

fn main() {}
