//@ run-fail
//@ check-run-results
//@ compile-flags: -C debug-assertions
//@ needs-subprocess

#![allow(arithmetic_overflow)]

fn main() {
    let _x = 200u8 + 200u8 + 200u8;
}
