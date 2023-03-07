// check-fail

#![feature(auto_traits)]
#![deny(where_clauses_object_safety)]

auto trait AutoTrait {}

trait Trait {
    fn static_lifetime_bound(&self) where Self: 'static {}

    fn arg_lifetime_bound<'a>(&self, _arg: &'a ()) where Self: 'a {}

    fn autotrait_bound(&self) where Self: AutoTrait {}
}

impl Trait for () {}

fn main() {
    let trait_object = &() as &dyn Trait;
    trait_object.static_lifetime_bound();
    trait_object.arg_lifetime_bound(&());
    trait_object.autotrait_bound(); //~ ERROR: the trait bound `dyn Trait: AutoTrait` is not satisfied
}
