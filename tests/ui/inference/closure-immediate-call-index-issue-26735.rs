//! Regression test for https://github.com/rust-lang/rust/issues/26735.
//! An immediate call should constrain closure inputs before its body is checked.

//@ check-pass

fn main() {
    let matrix = [[0.0; 2]; 2];
    let _ = (|i, j| matrix[i][j])(0, 1);
}
