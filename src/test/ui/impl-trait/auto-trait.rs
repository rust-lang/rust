// Tests that type alias impls traits do not leak auto-traits for
// the purposes of coherence checking
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
impl<T: Send> AnotherTrait for T {}

// This is in error, because we cannot assume that `OpaqueType: !Send`.
// (We treat opaque types as "foreign types" that could grow more impls
// in the future.)
impl AnotherTrait for D<OpaqueType> {
    //~^ ERROR conflicting implementations of trait `AnotherTrait` for type `D<impl OpaqueTrait>`
}

fn main() {}
