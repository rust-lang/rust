//! regression test for https://github.com/rust-lang/rust/issues/24779
//@ run-pass
fn main() {
    assert_eq!((|| || 42)()(), 42);
}
