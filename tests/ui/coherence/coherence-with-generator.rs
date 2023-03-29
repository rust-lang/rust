// Test that encountering closures during coherence does not cause issues.
#![feature(type_alias_impl_trait, generators)]
#![cfg_attr(specialized, feature(specialization))]
#![allow(incomplete_features)]

// revisions: stock specialized
// [specialized]check-pass

type OpaqueGenerator = impl Sized;
#[defines(OpaqueGenerator)]
fn defining_use() -> OpaqueGenerator {
    || {
        for i in 0..10 {
            yield i;
        }
    }
}

struct Wrapper<T>(T);
trait Trait {}
impl Trait for Wrapper<OpaqueGenerator> {}
impl<T: Sync> Trait for Wrapper<T> {}
//[stock]~^ ERROR conflicting implementations of trait `Trait` for type `Wrapper<OpaqueGenerator>`

fn main() {}
