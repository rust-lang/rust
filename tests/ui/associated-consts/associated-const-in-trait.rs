// #29924

trait Trait {
    const N: usize;
}

impl dyn Trait {
    //~^ ERROR the trait `Trait` cannot be made into an object [E0038]
    const fn n() -> usize { Self::N }
}

fn main() {}
