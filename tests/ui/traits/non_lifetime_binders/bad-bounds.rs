//@ edition: 2024

#![feature(non_lifetime_binders)]
#![expect(incomplete_features)]

fn produce() -> for<A: A<{ //~ ERROR: expected a type, found a trait
    //~^ ERROR: late-bound type parameter not allowed on trait object types
    //~| ERROR: expected trait, found type parameter `A`
    #[derive(Hash)]
    enum A {} //~ ERROR: missing generics for struct `produce::{constant#0}::A`
    struct A<A>; //~ ERROR: the name `A` is defined multiple times
                 //~^ ERROR: type parameter `A` is never used
    }>> Trait{ //~ ERROR: cannot find trait `Trait` in this scope
}

fn main() {}
