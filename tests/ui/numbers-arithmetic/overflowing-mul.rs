//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ regex-error-pattern: attempt to compute.*\*.* which would overflow
//@ needs-subprocess
//@ compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let x = 200u8 * 4;
}
