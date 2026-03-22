//@ run-fail
//@ regex-error-pattern: thread 'main'.*panicked
//@ regex-error-pattern: attempt to compute.*\+.* which would overflow
//@ compile-flags: -C debug-assertions
//@ needs-subprocess

#![allow(arithmetic_overflow)]

fn main() {
    let _x = u128::MAX + 100;
}
