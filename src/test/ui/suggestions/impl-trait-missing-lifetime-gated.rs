// edition:2021
// gate-test-anonymous_lifetime_in_impl_trait
// Verify the behaviour of `feature(anonymous_lifetime_in_impl_trait)`.

fn f(_: impl Iterator<Item = &'_ ()>) {}
//~^ ERROR anonymous lifetimes in `impl Trait` are unstable

fn g(x: impl Iterator<Item = &'_ ()>) -> Option<&'_ ()> { x.next() }
//~^ ERROR anonymous lifetimes in `impl Trait` are unstable
//~| ERROR missing lifetime specifier

// Anonymous lifetimes in async fn are already allowed.
// This is understood as `fn foo<'_1>(_: impl Iterator<Item = &'_1 ()>) {}`.
async fn h(_: impl Iterator<Item = &'_ ()>) {}

// Anonymous lifetimes in async fn are already allowed.
// But that lifetime does not participate in resolution.
async fn i(x: impl Iterator<Item = &'_ ()>) -> Option<&'_ ()> { x.next() }
//~^ ERROR missing lifetime specifier

fn main() {}
