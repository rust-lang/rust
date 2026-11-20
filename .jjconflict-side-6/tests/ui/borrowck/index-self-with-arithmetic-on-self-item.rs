//! Regression test for <https://github.com/rust-lang/rust/issues/29743>.
//! Test borrowck doesn't complain about using arithmetic with self item
//! as index.
//@ check-pass

fn main() {
    let mut i = [1, 2, 3];
    i[i[0]] = 0;
    i[i[0] - 1] = 0;
}
