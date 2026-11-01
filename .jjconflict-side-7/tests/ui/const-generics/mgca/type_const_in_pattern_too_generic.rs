#![feature(gca_min_const_items)]
#![expect(incomplete_features)]

trait Trait {
    #[rustc_always_gca]
    const ASSOC: usize;
}

fn test<T: Trait>() {
    if let <T as Trait>::ASSOC = 1 {}
    //~^ ERROR could not evaluate constant pattern
}

fn main() {}
