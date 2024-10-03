//@ edition:2021

trait Trait<const N: Trait = bar> {
    //~^ ERROR: cannot find value `bar` in this scope
    //~| ERROR: cycle detected when computing type of `Trait::N`
    //~| ERROR: the trait `Trait` cannot be made into an object
    //~| ERROR: the trait `Trait` cannot be made into an object
    //~| ERROR: the trait `Trait` cannot be made into an object
    //~| ERROR: trait objects must include the `dyn` keyword
    async fn a() {}
}

fn main() {}
