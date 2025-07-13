//@ run-pass

pub fn main() {
    assert_eq!((0 + 0u8) as char, '\0');
}

// https://github.com/rust-lang/rust/issues/9918
