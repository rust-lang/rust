pub trait Trait {
    type A;
}

pub type Alias = dyn Trait<A = Self::A>;
//~^ ERROR failed to resolve: `Self`

fn main() {}
