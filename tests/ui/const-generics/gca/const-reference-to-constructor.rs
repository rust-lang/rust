//@ check-pass
//@ compile-flags: -Znext-solver
//! https://github.com/rust-lang/rust/issues/162923
#![feature(gca_min_const_items)]
#![feature(gca_const_items)]
enum T<const N: u8 = { T::<0>::B as u8 }> {
    A = 2,
    B,
}
fn main() {}
