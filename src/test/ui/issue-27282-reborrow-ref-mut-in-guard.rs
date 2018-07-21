// Copyright 2018 The Rust Project Developers. See the COPYRIGHT
// file at the top-level directory of this distribution and at
// http://rust-lang.org/COPYRIGHT.
//
// Licensed under the Apache License, Version 2.0 <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT license
// <LICENSE-MIT or http://opensource.org/licenses/MIT>, at your
// option. This file may not be copied, modified, or distributed
// except according to those terms.

// Issue 27282: This is a variation on issue-27282-move-ref-mut-into-guard.rs
//
// It reborrows instead of moving the `ref mut` pattern borrow. This
// means that our conservative check for mutation in guards will
// reject it. But I want to make sure that we continue to reject it
// (under NLL) even when that conservaive check goes away.

// compile-flags: -Z disable-ast-check-for-mutation-in-guard

#![feature(nll)]

fn main() {
    let mut b = &mut true;
    match b {
        &mut false => {},
        ref mut r if { (|| { let bar = &mut *r; **bar = false; })();
        //~^ ERROR cannot borrow `r` as mutable, as it is immutable for the pattern guard
                             false } => { &mut *r; },
        &mut true => { println!("You might think we should get here"); },
        _ => panic!("surely we could never get here, since rustc warns it is unreachable."),
    }
}
