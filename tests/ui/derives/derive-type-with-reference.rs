//! Regression test for https://github.com/rust-lang/rust/issues/15689.
//@ run-pass

#[derive(PartialEq, Debug, Clone)]
enum Test<'a> {
    Slice(&'a isize),
}

fn main() {
    assert_eq!(Test::Slice(&1), Test::Slice(&1))
}
