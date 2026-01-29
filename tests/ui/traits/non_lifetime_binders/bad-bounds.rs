//@ edition: 2024

#![feature(non_lifetime_binders)]
#![expect(incomplete_features)]

fn produce() -> for<A: A<{ //~ ERROR expected trait, found type parameter `A`
    //~^ ERROR bounds cannot be used in this context
    //~^^ ERROR late-bound type parameter not allowed on trait object types
    //~^^^ ERROR expected a type, found a trait
    #[derive(Hash)] //~ ERROR missing generics for struct `produce::{constant#0}::A`
    enum A {}
    struct A<A>; //~ ERROR the name `A` is defined multiple times
}>> Trait {} //~ ERROR cannot find trait `Trait` in this scope

fn main() {}
