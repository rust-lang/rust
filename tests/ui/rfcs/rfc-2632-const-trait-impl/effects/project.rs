// revisions: never_const_super always_const_super maybe_const_super
//[maybe_const_super] check-pass

#![feature(const_trait_impl, effects)]

#[const_trait]
pub trait Owo<X = <Self as ~const Uwu>::T> {}

// This fails because `~const Uwu` doesn't imply (non-const) `Uwu`.
#[cfg(never_const_super)]
#[const_trait]
pub trait Uwu: Owo { type T; }
//[never_const_super]~^ ERROR the trait bound `Self: Uwu` is not satisfied
//[never_const_super]~| ERROR the trait bound `Self: Uwu` is not satisfied
//[never_const_super]~| ERROR the trait bound `Self: Uwu` is not satisfied
//[never_const_super]~| ERROR the trait bound `Self: Uwu` is not satisfied
//[never_const_super]~| ERROR the trait bound `Self: Uwu` is not satisfied

// FIXME(effects): Shouldn't this pass?
#[cfg(always_const_super)]
#[const_trait]
pub trait Uwu: const Owo { type T; }
//[always_const_super]~^ ERROR the trait bound `Self: const Uwu` is not satisfied
//[always_const_super]~| ERROR the trait bound `Self: const Uwu` is not satisfied
//[always_const_super]~| ERROR the trait bound `Self: const Uwu` is not satisfied
//[always_const_super]~| ERROR the trait bound `Self: const Uwu` is not satisfied
//[always_const_super]~| ERROR the trait bound `Self: const Uwu` is not satisfied

#[cfg(maybe_const_super)]
#[const_trait]
pub trait Uwu: ~const Owo { type T; }

fn main() {}
