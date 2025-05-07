#![feature(type_alias_impl_trait)]
#![cfg_attr(specialized, feature(specialization))]
#![allow(incomplete_features)]

//@ revisions: stock specialized
//@ [specialized]check-pass

trait OpaqueTrait {}
impl<T> OpaqueTrait for T {}
type OpaqueType = impl OpaqueTrait;
#[define_opaque(OpaqueType)]
fn mk_opaque() -> OpaqueType {
    || 0
}
trait AnotherTrait {}
impl<T: Send> AnotherTrait for T {}
impl AnotherTrait for OpaqueType {}
//[stock]~^ ERROR conflicting implementations of trait `AnotherTrait`

fn main() {}
