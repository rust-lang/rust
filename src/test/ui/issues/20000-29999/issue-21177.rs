trait Trait {
    type A;
    type B;
}

fn foo<T: Trait<A = T::B>>() { }
//~^ ERROR cycle detected

fn main() { }
