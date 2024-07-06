//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass
//@ check-run-results
//@ normalize-stdout-test: "\d+" -> "<..>"

// https://github.com/rust-lang/rust/issues/107975#issuecomment-1430704499

#![feature(strict_provenance)]

use std::ptr::addr_of;

fn main() {
    let a = {
        let v = 0;
        addr_of!(v).addr()
    };
    let b = {
        let v = 0;
        addr_of!(v).addr()
    };

    println!("{}", a == b); // prints false
    println!("{a}"); // or b
    println!("{}", a == b); // prints true
}
