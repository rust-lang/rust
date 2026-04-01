#![feature(transmutability)]
#![feature(type_alias_impl_trait)]
trait OpaqueTrait {}
type OpaqueType = impl OpaqueTrait;
//~^ ERROR unconstrained opaque type
trait AnotherTrait {}
impl<T: std::mem::TransmuteFrom<(), ()>> AnotherTrait for T {}
//~^ ERROR type provided when a constant was expected
impl AnotherTrait for OpaqueType {}
//~^ ERROR conflicting implementations of trait `AnotherTrait`
pub fn main() {}
