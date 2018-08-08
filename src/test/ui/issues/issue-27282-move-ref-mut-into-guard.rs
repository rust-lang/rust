// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 27282: Example 1: This sidesteps the AST checks disallowing
// mutable borrows in match guards by hiding the mutable borrow in a
// guard behind a move (of the ref mut pattern id) within a closure.
//
// This example is not rejected by AST borrowck (and then reliably
// segfaults when executed).

#![feature(nll)]

fn main() {
    match Some(&4) {
        None => {},
        ref mut foo
            if { (|| { let bar = foo; bar.take() })(); false } => {},
        //~^ ERROR cannot move out of borrowed content [E0507]
        Some(s) => std::process::exit(*s),
    }
}
