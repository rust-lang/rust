//@ run-pass
//! regression test for <https://github.com/rust-lang/rust/issues/3091>

pub fn main() {
    let x = 1;
    let y = 1;
    assert_eq!(&x, &y);
}
