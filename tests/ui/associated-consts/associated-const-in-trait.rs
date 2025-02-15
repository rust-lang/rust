// #29924

trait Trait {
    const N: usize;
}

impl dyn Trait {
    //~^ ERROR the trait `Trait` is not dyn compatible [E0038]
    const fn n() -> usize { Self::N }
    //~^ ERROR the trait `Trait` is not dyn compatible [E0038]
    //~| ERROR the trait `Trait` is not dyn compatible
}

fn main() {}
