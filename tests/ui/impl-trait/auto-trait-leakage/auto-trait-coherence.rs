// Tests that type alias impls traits do not leak auto-traits for
// the purposes of coherence checking
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
impl<T: Send> AnotherTrait for T {}

// This is in error, because we cannot assume that `OpaqueType: !Send`.
// (We treat opaque types as "foreign types" that could grow more impls
// in the future.)
impl AnotherTrait for D<OpaqueType> {
    //~^ ERROR conflicting implementations of trait `AnotherTrait` for type `D<_>`
}

fn main() {}
