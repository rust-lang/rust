// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 46112: An extern crate pub re-exporting libcore was causing
// paths rooted from `std` to be misrendered in the diagnostic output.

// ignore-windows
// aux-build:xcrate_issue_43189_a.rs
// aux-build:xcrate_issue_43189_b.rs

extern crate xcrate_issue_43189_b;
fn main() {
    ().a();
    //~^ ERROR no method named `a` found for type `()` in the current scope [E0599]
}
