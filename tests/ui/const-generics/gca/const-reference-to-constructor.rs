//@ check-pass
//@ compile-flags: -Znext-solver
//! https://github.com/rust-lang/rust/issues/162923
#![feature(min_generic_const_args)]
#![feature(generic_const_args)]
enum T<const N: u8 = { T::<0>::B as u8 }> {
    A = 2,
    B,
}
fn main() {}
