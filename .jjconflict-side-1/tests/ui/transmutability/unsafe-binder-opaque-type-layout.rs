//! Regression test for https://github.com/rust-lang/rust/issues/141400.

#![feature(unsafe_binders)]
#![feature(transmutability)]
#![feature(type_alias_impl_trait)]
#![allow(incomplete_features)]

trait OpaqueTrait {}
type OpaqueType = unsafe<> impl OpaqueTrait;
//~^ ERROR the trait bound `OpaqueType::{opaque#0}: Copy` is not satisfied
//~| ERROR unconstrained opaque type
trait AnotherTrait {}
impl<T: std::mem::TransmuteFrom<()>> AnotherTrait for T {}
impl AnotherTrait for OpaqueType {}
//~^ ERROR conflicting implementations of trait `AnotherTrait`

pub fn main() {}
