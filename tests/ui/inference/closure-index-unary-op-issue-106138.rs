//! Regression test for https://github.com/rust-lang/rust/issues/106138.
//! Index types beneath unary operators should be inferred from later closure calls.

//@ check-pass

fn bools(x: &Vec<bool>) {
    let logical_not = |i, values: &Vec<bool>| !values[i];
    let _ = logical_not(0, x);
}

fn ints(x: &Vec<isize>) {
    let negate = |i, values: &Vec<isize>| -values[i];
    let _ = negate(0, x);
}

fn main() {
    bools(&vec![true]);
    ints(&vec![1]);
}
