#![feature(const_trait_impl)]
//@ edition: 2021

const trait Trait {}

fn main() {
    let _: &dyn const Trait; //~ ERROR const trait bounds are not allowed in trait object types
    let _: &dyn [const] Trait; //~ ERROR `[const]` is not allowed here
}

// Regression test for issue #119525.
trait NonConst {}
const fn handle(_: &dyn const NonConst) {}
//~^ ERROR const trait bounds are not allowed in trait object types
//~| ERROR `const` can only be applied to `const` traits
const fn take(_: &dyn [const] NonConst) {}
//~^ ERROR `[const]` is not allowed here
//~| ERROR `[const]` can only be applied to `const` traits
