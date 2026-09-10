//@ compile-flags: -Znext-solver

// Regression test for trait-system-refactor-initiative#242
// We used to drop all subsequent obligations when one obligation overflows
// in fulfillment. It means we don't really prove all obligations even if
// fulfillment returns no error.
//
// We now eagerly abort on the first overflowed obligation.
//
// FIXME: this probably should compile and we shall fix duplicate
// uses of opaques.

#![feature(type_alias_impl_trait)]
type Tait<'a> = impl Sized;

fn prove()
where
    for<'a> Tait<'a>: Sized,
{}

#[define_opaque(Tait)]
fn foo<'a>() -> &'a Tait<'a> {
    prove();
    //~^ ERROR: overflow evaluating the requirement `for<'a> Tait<'a>: Sized`
    &()
}
fn main() {}
