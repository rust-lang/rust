#![feature(const_trait_impl)]
// FIXME(const_trait_impl) add effects
//@ edition: 2021

#[const_trait]
trait Trait {}

fn main() {
    let _: &dyn const Trait; //~ ERROR const trait bounds are not allowed in trait object types
    let _: &dyn [const] Trait; //~ ERROR `[const]` is not allowed here
}

// Regression test for issue #119525.
trait NonConst {}
const fn handle(_: &dyn const NonConst) {}
//~^ ERROR const trait bounds are not allowed in trait object types
const fn take(_: &dyn [const] NonConst) {}
//~^ ERROR `[const]` is not allowed here
