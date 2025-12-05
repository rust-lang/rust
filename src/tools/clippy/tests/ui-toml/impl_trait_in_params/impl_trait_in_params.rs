//! As avoid-breaking-exported-api is `false`, nothing here should lint
#![warn(clippy::impl_trait_in_params)]
#![no_main]
//@no-rustfix

pub trait Trait {}

trait Private {
    fn t(_: impl Trait);
    fn tt<T: Trait>(_: T);
}

pub trait Public {
    fn t(_: impl Trait); //~ ERROR: `impl Trait` used as a function parameter
    fn tt<T: Trait>(_: T);
}
