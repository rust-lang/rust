//@ edition:2021

trait Trait<const N: Trait = bar> {
    //~^ ERROR: cannot find value `bar` in this scope
    //~| ERROR: cycle detected when computing predicates of `Trait`
    async fn a() {}
}

fn main() {}
