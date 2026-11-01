//@ run-rustfix
#![expect(incomplete_features)]
#![feature(explicit_tail_calls)]
#![allow(unused)]
use std::num::Wrapping;
use std::ops::{Not, Add, BitXorAssign};

// built-ins and overloaded operators are handled differently

fn f(a: u64, b: u64) -> u64 {
    become a + b; //~ error: `become` does not support operators
}

fn g(a: String, b: &str) -> String {
    become a + b; //~ error: `become` does not support operators
}

fn h(x: u64) -> u64 {
    become !x; //~ error: `become` does not support operators
}

fn i_do_not_know_any_more_letters(x: Wrapping<u32>) -> Wrapping<u32> {
    become !x; //~ error: `become` does not support operators
}

fn builtin_index(x: &[u8], i: usize) -> u8 {
    become x[i] //~ error: `become` does not support operators
}

// FIXME(explicit_tail_calls): overloaded index is represented like `[&]*x.index(i)`,
//                             and so need additional handling

fn a(a: &mut u8, _: u8) {
    become *a ^= 1; //~ error: `become` does not support operators
}

fn b(b: &mut Wrapping<u8>, _: u8) {
    become *b ^= 1; //~ error: `become` does not support operators
}


fn main() {}
