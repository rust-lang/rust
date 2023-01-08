// check-fail
// aux-build: autotrait.rs

#![deny(where_clauses_object_safety)]

extern crate autotrait;

use autotrait::AutoTrait as NonlocalAutoTrait;

unsafe trait UnsafeTrait {}

trait Trait {
    fn static_lifetime_bound(&self) where Self: 'static {}

    fn arg_lifetime_bound<'a>(&self, _arg: &'a ()) where Self: 'a {}

    fn unsafe_trait_bound(&self) where Self: UnsafeTrait {}

    fn nonlocal_autotrait_bound(&self) where Self: NonlocalAutoTrait {}
}

impl Trait for () {}

fn main() {
    let trait_object = &() as &dyn Trait;
    trait_object.static_lifetime_bound();
    trait_object.arg_lifetime_bound(&());
    trait_object.unsafe_trait_bound(); //~ ERROR: the trait bound `dyn Trait: UnsafeTrait` is not satisfied
    trait_object.nonlocal_autotrait_bound(); //~ ERROR: the trait bound `dyn Trait: AutoTrait` is not satisfied
}
