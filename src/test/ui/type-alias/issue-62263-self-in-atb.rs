pub trait Trait {
    type A;
}

pub type Alias = dyn Trait<A = Self::A>;
//~^ ERROR failed to resolve: use of undeclared type or module `Self` [E0433]

fn main() {}
