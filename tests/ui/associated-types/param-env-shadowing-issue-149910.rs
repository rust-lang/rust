trait Trait {
    type Assoc;
}

impl<T> Trait for T {
    type Assoc = T;
}

fn foo<T: Trait>(x: T) -> T::Assoc {
    x
    //~^ ERROR mismatched types
}

fn main() {}
