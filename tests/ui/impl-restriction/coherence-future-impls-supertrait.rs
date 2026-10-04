//@ check-pass
//@ revisions: classic coherence next
//@[classic] compile-flags: -Znext-solver=no
//@[coherence] compile-flags: -Znext-solver=coherence
//@[next] compile-flags: -Znext-solver=globally
//@ aux-build: coherence-future-impls.rs

extern crate coherence_future_impls as upstream;

struct Thing;

trait LocalTrait {}

impl<T: upstream::RestrictedWithSupertrait> LocalTrait for T {}

// Although `RestrictedWithSupertrait` permits future upstream impls,
// its `OpenSupertrait` requirement cannot hold for this local type.
impl LocalTrait for Thing {}

fn main() {}
