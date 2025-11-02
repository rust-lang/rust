// https://github.com/rust-lang/rust/issues/9918
//@ run-pass

pub fn main() {
    assert_eq!((0 + 0u8) as char, '\0');
}
