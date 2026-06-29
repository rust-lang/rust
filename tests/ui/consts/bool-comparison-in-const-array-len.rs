//! Regression test for https://github.com/rust-lang/rust/issues/39548

//@ run-pass
type Array = [(); ((1 < 2) == false) as usize];

fn main() {
    let _: Array = [];
}
