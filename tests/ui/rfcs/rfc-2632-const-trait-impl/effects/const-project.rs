// revisions: never_const_super always_const_super maybe_const_super

#![feature(const_trait_impl, effects)]
#![feature(associated_const_equality, generic_const_exprs)]
#![allow(incomplete_features)]

// FIXME(effects): Shouldn't `{always,maybe}_const_super` pass? Also, why does
// the diagnostic message and location differ from `effects/project.rs`?

#[const_trait]
pub trait Owo<const X: u32 = { <Self as ~const Uwu>::K }> {}
//~^ ERROR the trait bound `Self: ~const Uwu` is not satisfied

#[cfg(never_const_super)]
#[const_trait]
pub trait Uwu: Owo { const K: u32; }

#[cfg(always_const_super)]
#[const_trait]
pub trait Uwu: const Owo { const K: u32; }

#[cfg(maybe_const_super)]
#[const_trait]
pub trait Uwu: ~const Owo { const K: u32; }

fn main() {}
