//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ error-pattern: attempt to add with overflow
//@ compile-flags: -C debug-assertions
//@ needs-subprocess

#![allow(arithmetic_overflow)]

fn main() {
    let _x = 200u8 + 200u8 + 200u8;
}
