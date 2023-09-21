//! As avoid-breaking-exported-api is `true`, nothing here should lint
#![warn(clippy::impl_trait_in_params)]
#![no_main]

trait Trait {}

trait T {
    fn t(_: impl Trait);
    fn tt<T: Trait>(_: T);
}
