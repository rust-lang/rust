//@ revisions: normal force
//@ edition: 2024
//@ aux-crate: force_unstable=force_unstable.rs
//@[force] compile-flags: -Zforce-unstable-if-unmarked

#![feature(rustc_private)]

// Regression test for <https://github.com/rust-lang/rust/issues/152692>.
//
// When building a crate with `-Zforce-unstable-if-unmarked` (e.g. the compiler or stdlib),
// it's unhelpful to mention that a not-implemented trait is unstable, because that will
// be true of every local and foreign trait that isn't explicitly marked stable.

trait LocalTrait {}

fn use_local_trait(_: impl LocalTrait) {}
//~^ NOTE required by a bound in `use_local_trait`
//~| NOTE required by this bound in `use_local_trait`

fn use_foreign_trait(_: impl force_unstable::ForeignTrait) {}
//~^ NOTE required by a bound in `use_foreign_trait`
//~| NOTE required by this bound in `use_foreign_trait`

fn main() {
    use_local_trait(());
    //~^ ERROR the trait bound `(): LocalTrait` is not satisfied
    //[normal]~| NOTE the trait `LocalTrait` is not implemented for `()`
    //[force]~| NOTE the trait `LocalTrait` is not implemented for `()`
    //~| NOTE required by a bound introduced by this call

    use_foreign_trait(());
    //~^ ERROR the trait bound `(): ForeignTrait` is not satisfied
    //[normal]~| NOTE the nightly-only, unstable trait `ForeignTrait` is not implemented for `()`
    //[force]~| NOTE the trait `ForeignTrait` is not implemented for `()`
    //~| NOTE required by a bound introduced by this call
}
