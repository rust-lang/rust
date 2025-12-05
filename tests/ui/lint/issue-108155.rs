//@ check-pass
// check that `deref_into_dyn_supertrait` doesn't cause ICE by eagerly converting
// a cancelled lint

#![allow(deref_into_dyn_supertrait)]

trait Trait {}
impl std::ops::Deref for dyn Trait + Send + Sync {
    type Target = dyn Trait;
    fn deref(&self) -> &Self::Target {
        self
    }
}

fn main() {}
