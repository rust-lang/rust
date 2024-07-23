pub trait Trait {
    type A;
}

pub type Alias = dyn Trait<A = Self::A>;
//~^ ERROR cannot find item `Self` in this scope

fn main() {}
