// Copyright 2014 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// ignore-msvc FIXME #31306

// note that these aux-build directives must be in this order
// aux-build:svh-a-base.rs
// aux-build:svh-b.rs
// aux-build:svh-a-change-type-static.rs

extern crate a;
extern crate b; //~ ERROR: found possibly newer version of crate `a` which `b` depends on
//~| NOTE: perhaps that crate needs to be recompiled
//~| NOTE: crate `a` path #1:
//~| NOTE: crate `b` path #1:

fn main() {
    b::foo()
}
