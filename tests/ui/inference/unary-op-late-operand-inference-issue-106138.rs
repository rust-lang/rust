//! Regression test for https://github.com/rust-lang/rust/issues/106138.
//! Unary operators should allow their operand types to be inferred by later constraints.

//@ check-pass

use std::ops::Not;

fn not_index(x: &Vec<bool>) {
    let closure = |i, a: &Vec<bool>| !a[i];
    let _ = closure(0, x);
}

fn neg_index(x: &Vec<i32>) {
    let closure = |i, a: &Vec<i32>| -a[i];
    let _ = closure(0, x);
}

#[derive(Copy, Clone, Default)]
struct Input;

struct Output;

impl Not for Input {
    type Output = Output;

    fn not(self) -> Self::Output {
        Output
    }
}

fn output_differs_from_operand() {
    let input = Default::default();
    let output = !input;
    let _: Input = input;
    let _: Output = output;
}

fn output_expectation_differs_from_operand() {
    let input = Default::default();
    let _: Output = !input;
    let _: Input = input;
}

fn main() {
    not_index(&vec![true]);
    neg_index(&vec![1]);
    output_differs_from_operand();
    output_expectation_differs_from_operand();
}
