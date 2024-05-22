//@ known-bug: #124207
#![feature(transmutability)]
#![feature(type_alias_impl_trait)]
trait OpaqueTrait {}
type OpaqueType = impl OpaqueTrait;
trait AnotherTrait {}
impl<T: std::mem::BikeshedIntrinsicFrom<(), ()>> AnotherTrait for T {}
impl AnotherTrait for OpaqueType {}
pub fn main() {}
