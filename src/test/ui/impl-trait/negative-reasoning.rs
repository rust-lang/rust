// Tests that we cannot assume that an opaque type does *not* implement some
// other trait
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

trait OpaqueTrait {}
impl<T> OpaqueTrait for T {}
type OpaqueType = impl OpaqueTrait;
fn mk_opaque() -> OpaqueType {
    ()
}

#[derive(Debug)]
struct D<T>(T);

trait AnotherTrait {}
impl<T: std::fmt::Debug> AnotherTrait for T {}

// This is in error, because we cannot assume that `OpaqueType: !Debug`
impl AnotherTrait for D<OpaqueType> {
    //~^ ERROR conflicting implementations of trait `AnotherTrait` for type `D<impl OpaqueTrait>`
}

fn main() {}
