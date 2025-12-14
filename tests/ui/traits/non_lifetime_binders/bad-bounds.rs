//@ edition 2024

#![feature(non_lifetime_binders)]
fn produce() -> for<A: A<{ //~ ERROR expected trait, found type parameter `A`
    //~^ ERROR bounds cannot be used in this context
    #[derive(Hash)]
    enum A {}
    struct A<A>; //~ ERROR the name `A` is defined multiple times
}>> Trait {} //~ ERROR cannot find trait `Trait` in this scope

fn main() {}
