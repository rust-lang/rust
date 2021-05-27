// run-pass
// edition:2021
// compile-flags: -Zunstable-options

// regression test for https://github.com/rust-lang/rust/pull/85678

#![feature(assert_matches)]

fn main() {
    assert!(matches!((), ()));
    assert_matches!((), ());
}
