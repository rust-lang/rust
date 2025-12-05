#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

struct Combination<const STRATEGIES: usize>;

impl<const STRATEGIES: usize> Combination<STRATEGIES> {
    fn and<M>(self) -> Combination<{ STRATEGIES + 1 }> {
        Combination
    }
}

pub fn main() {
    Combination::<0>.and::<_>().and::<_>();
    //~^ ERROR: type annotations needed
}
