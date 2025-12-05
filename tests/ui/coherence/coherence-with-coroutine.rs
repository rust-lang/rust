// Test that encountering closures during coherence does not cause issues.
#![feature(type_alias_impl_trait, coroutines)]
#![cfg_attr(specialized, feature(specialization))]
#![allow(incomplete_features)]

//@ revisions: stock specialized
//@ [specialized]check-pass

type OpaqueCoroutine = impl Sized;
#[define_opaque(OpaqueCoroutine)]
fn defining_use() -> OpaqueCoroutine {
    #[coroutine]
    || {
        for i in 0..10 {
            yield i;
        }
    }
}

struct Wrapper<T>(T);
trait Trait {}
impl Trait for Wrapper<OpaqueCoroutine> {}
impl<T: Sync> Trait for Wrapper<T> {}
//[stock]~^ ERROR conflicting implementations of trait `Trait` for type `Wrapper<OpaqueCoroutine>`

fn main() {}
