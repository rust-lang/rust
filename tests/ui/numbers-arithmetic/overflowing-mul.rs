//@ run-fail
//@ check-run-results
//@ needs-subprocess
//@ compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let x = 200u8 * 4;
}
