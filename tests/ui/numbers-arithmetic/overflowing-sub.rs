//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ error-pattern: attempt to subtract with overflow
//@ needs-subprocess
//@ compile-flags: -C debug-assertions

#![allow(arithmetic_overflow)]

fn main() {
    let _x = 42u8 - (42u8 + 1);
}
