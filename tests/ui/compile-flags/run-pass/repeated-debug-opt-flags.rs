//! regression test for https://github.com/rust-lang/rust/issues/24945
//@ run-pass
// This test is just checking that we continue to accept `-g -g -O -O`
// as options to the compiler.

//@ compile-flags:-g -g -O -O

fn main() {
    assert_eq!(1, 1);
}
