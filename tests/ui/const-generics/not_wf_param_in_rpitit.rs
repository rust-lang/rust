//@ edition:2021

trait Trait<const N: dyn Trait = bar> {
    //~^ ERROR: cannot find value `bar` in this scope
    //~| ERROR: cycle detected when computing type of `Trait::N`
    //~| ERROR: the trait `Trait` cannot be made into an object
    //~| ERROR: the trait `Trait` cannot be made into an object
    //~| ERROR: the trait `Trait` cannot be made into an object
    async fn a() {}
}

fn main() {}
