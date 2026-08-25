//! regression test for <https://github.com/rust-lang/rust/issues/50471>
//@ check-pass

fn main() {
    assert!({ false });

    assert!(r"\u{41}" == "A");

    assert!(r"\u{".is_empty());
}
