// Tests that we cannot assume that an opaque type does *not* implement some
// other trait
#![feature(type_alias_impl_trait)]

trait OpaqueTrait {}
impl<T> OpaqueTrait for T {}
type OpaqueType = impl OpaqueTrait;
#[define_opaque(OpaqueType)]
fn mk_opaque() -> OpaqueType {
    ()
}

#[derive(Debug)]
struct D<T>(T);

trait AnotherTrait {}
impl<T: std::fmt::Debug> AnotherTrait for T {}

// This is in error, because we cannot assume that `OpaqueType: !Debug`
impl AnotherTrait for D<OpaqueType> {
    //~^ ERROR conflicting implementations of trait `AnotherTrait` for type `D<_>`
}

fn main() {}
