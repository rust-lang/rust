//! Regression test for https://github.com/rust-lang/rust/issues/30615
//! This test confirms that valid code casting a thin pointer to a fat pointer does not cause ICE
//! and compiles.

//@ run-pass
fn main() {
    &0u8 as *const u8 as *const dyn PartialEq<u8>;
    &[0u8] as *const [u8; 1] as *const [u8];
}
