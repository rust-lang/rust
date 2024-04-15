// A regression test for #124207.
//
// This previous caused an ICE in the old solver.
#![feature(transmutability)]
#![feature(type_alias_impl_trait)]
trait OpaqueTrait {}
type OpaqueType = impl OpaqueTrait;
//~^ ERROR unconstrained opaque type
trait AnotherTrait {}
impl<T: std::mem::BikeshedIntrinsicFrom<(), ()>> AnotherTrait for T {}
//~^ ERROR type provided when a constant was expected
impl AnotherTrait for OpaqueType {}
//~^ ERROR conflicting implementations of trait `AnotherTrait`
pub fn main() {}
