// Test that `const unsafe trait` and `const unsafe auto trait` works.

//@ compile-flags: -Zparse-crate-root-only
//@ check-pass

#![feature(const_trait_impl)]
#![feature(auto_traits)]

pub const unsafe trait Owo {}
const unsafe trait OwO {}
pub const unsafe auto trait UwU {}
const unsafe auto trait Uwu {}

fn main() {}
