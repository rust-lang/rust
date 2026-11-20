//@ edition: 2024

#![feature(non_lifetime_binders)]
#![expect(incomplete_features)]

fn produce() -> for<A: A<{ //~ ERROR expected trait, found type parameter `A`
    //~^ ERROR bounds cannot be used in this context
    #[derive(Hash)]
    enum B {}
    struct A<A>;
}>> Trait {} //~ ERROR cannot find trait `Trait` in this scope

fn main() {}
