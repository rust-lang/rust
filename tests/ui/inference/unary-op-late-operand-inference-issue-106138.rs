//! Regression test for https://github.com/rust-lang/rust/issues/106138.
//! Unary operators should allow their operand types to be inferred by later constraints.

//@ check-pass

use std::ops::Not;

// Indexing is a place projection, so `a[i]` autoderefs `a` and yields a `bool`.
// `!` is applied to that value: as a value operator it does not autoderef its
// operand itself, unlike the place projection `a[i]`.
fn not_index(x: &Vec<bool>) {
    let closure = |i, a: &Vec<bool>| !a[i];
    let _ = closure(0, x);
}

// Likewise, the `-` is applied to the `i32` produced by `a[i]`.
fn neg_index(x: &Vec<i32>) {
    let closure = |i, a: &Vec<i32>| -a[i];
    let _ = closure(0, x);
}

// Since the unary operators do not autoderef, these use `impl Not for &u32` and
// `impl Neg for &i32` rather than dereferencing to `u32`/`i32` first.
fn not_reference(x: &u32) -> u32 {
    !x
}

fn neg_reference(x: &i32) -> i32 {
    -x
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
    let _ = not_reference(&1);
    let _ = neg_reference(&1);
    output_differs_from_operand();
    output_expectation_differs_from_operand();
}
