//@ check-pass
// This should not ICE

// https://github.com/rust-lang/rust/issues/47639
pub fn test() {
    macro_rules! foo {
        () => ()
    }
}
