//@ edition: 2021

// Regression test for <https://github.com/rust-lang/rust/issues/118904>.
// `assert!(true)` and `assert!(!false)` should have similar coverage spans.

fn main() {
    assert!(true);
    assert!(!false);
    assert!(!!true);
    assert!(!!!false);
}
