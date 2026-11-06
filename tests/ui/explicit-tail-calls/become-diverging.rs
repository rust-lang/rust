//@ check-pass
//@ compile-flags: -W unreachable-code
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]

fn callee() -> ! {
    loop {}
}

fn caller() -> ! {
    become callee();
}

fn main() {}
