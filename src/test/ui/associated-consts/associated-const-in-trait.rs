// #29924

#![feature(const_fn, associated_consts)]

trait Trait {
    const N: usize;
}

impl dyn Trait {
    //~^ ERROR the trait `Trait` cannot be made into an object [E0038]
    const fn n() -> usize { Self::N }
}

fn main() {}
