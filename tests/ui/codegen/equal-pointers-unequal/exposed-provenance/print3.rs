//@ known-bug: #107975
//@ compile-flags: -Copt-level=2
//@ run-pass
//@ check-run-results
//@ normalize-stdout-test: "\d+" -> "<..>"

// https://github.com/rust-lang/rust/issues/107975#issuecomment-1430704499

#![feature(exposed_provenance)]

use std::ptr::addr_of;

fn main() {
    let a = {
        let v = 0;
        addr_of!(v).expose_provenance()
    };
    let b = {
        let v = 0;
        addr_of!(v).expose_provenance()
    };

    println!("{}", a == b); // false
    println!("{}", a == b); // false
    let c = a;
    println!("{} {} {}", a == b, a == c, b == c); // false true false
    println!("{a} {b}");
    println!("{} {} {}", a == b, a == c, b == c); // true true true
}
