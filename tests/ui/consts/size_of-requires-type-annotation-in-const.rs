//! Regression test for <https://github.com/rust-lang/rust/issues/47486>
fn main() {
    () < std::mem::size_of::<_>(); //~ ERROR: mismatched types
    [0u8; std::mem::size_of::<_>()]; //~ ERROR: type annotations needed
}
