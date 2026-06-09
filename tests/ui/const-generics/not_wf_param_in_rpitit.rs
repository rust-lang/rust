//@ edition:2021

trait Trait<const N: dyn Trait = bar> {
    //~^ ERROR: cannot find value `bar` in this scope
    //~| ERROR: cycle detected when computing type of `Trait::N`
    async fn a() {}
}

fn main() {}
