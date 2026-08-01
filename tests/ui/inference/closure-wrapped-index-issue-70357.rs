//! Regression test for https://github.com/rust-lang/rust/issues/70357.
//! Bounds on a wrapped closure should constrain its inputs before its body is checked.

//@ check-pass

fn take_callback(_: Option<impl Fn(usize, usize) -> usize>) {}

fn main() {
    let array = vec![vec![3; 10]; 10];
    take_callback(Some(|i, j| array[i][j]));
}
