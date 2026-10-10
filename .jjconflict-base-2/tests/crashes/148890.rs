//@ known-bug: #148890
impl std::ops::Neg for u128 {}

fn foo(-128..=127: u128) {}

fn main() {}
