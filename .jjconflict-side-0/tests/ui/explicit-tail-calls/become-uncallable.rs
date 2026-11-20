//@ run-rustfix
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]
#![allow(unused)]

fn f() -> u64 {
    become 1; //~ error: `become` requires a function call
}

fn g() {
    become { h() }; //~ error: `become` requires a function call
}

fn h() {
    become *&g(); //~ error: `become` requires a function call
}

fn main() {}
