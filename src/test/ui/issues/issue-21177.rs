trait Trait {
    type A;
    type B;
}

fn foo<T: Trait<A = T::B>>() { }
//~^ ERROR cycle detected
//~| ERROR associated type `B` not found for `T`

fn main() { }
