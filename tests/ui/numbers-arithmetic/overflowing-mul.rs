//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ error-pattern: attempt to multiply with overflow
//@ needs-subprocess
//@ compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let x = 200u8 * 4;
}
