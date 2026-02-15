//@ edition: 2024

#![feature(non_lifetime_binders)]
#![expect(incomplete_features)]

fn produce() -> for<A: A<{ //~ ERROR expected a type, found a trait
    //~^ ERROR bounds cannot be used in this context
    //~^^ ERROR late-bound type parameter not allowed on trait object types
    #[derive(Hash)]
    enum A {}
    struct A<A>;
}>> Trait {} //~ ERROR cannot find trait `Trait` in this scope

fn main() {}
