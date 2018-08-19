// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 27282: Example 2: This sidesteps the AST checks disallowing
// mutable borrows in match guards by hiding the mutable borrow in a
// guard behind a move (of the mutably borrowed match input) within a
// closure.
//
// This example is not rejected by AST borrowck (and then reliably
// reaches the panic code when executed, despite the compiler warning
// about that match arm being unreachable.

#![feature(nll)]

fn main() {
    let b = &mut true;
    match b {
        &mut false => {},
        _ if { (|| { let bar = b; *bar = false; })();
                     //~^ ERROR cannot move out of `b` because it is borrowed [E0505]
                     false } => { },
        &mut true => { println!("You might think we should get here"); },
        //~^ ERROR use of moved value: `*b` [E0382]
        _ => panic!("surely we could never get here, since rustc warns it is unreachable."),
    }
}
