// Reported in <https://github.com/rust-lang/rust/pull/148434#issuecomment-3621280430>.
//@ check-pass

#![feature(const_trait_impl)]

const unsafe impl Trait for () {}

const unsafe trait Trait {}

fn main() {}
