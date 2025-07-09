//@ run-fail
//@ check-run-results
//@ needs-subprocess
//@ compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let _x = 42u8 - (42u8 + 1);
}
