//! Regression test for https://github.com/rust-lang/rust/issues/12677

//@ run-pass

fn main() {
    let s = "Hello";
    let first = s.bytes();
    let second = first.clone();

    assert_eq!(first.collect::<Vec<u8>>(), second.collect::<Vec<u8>>())
}
