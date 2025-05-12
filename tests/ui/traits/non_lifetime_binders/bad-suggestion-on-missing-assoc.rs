#![feature(generic_const_exprs)]
//~^ WARN the feature `generic_const_exprs` is incomplete
#![feature(non_lifetime_binders)]
//~^ WARN the feature `non_lifetime_binders` is incomplete

// Test for <https://github.com/rust-lang/rust/issues/115497>,
// which originally relied on associated_type_bounds, but was
// minimized away from that.

trait TraitA {
    type AsA;
}
trait TraitB {
    type AsB;
}
trait TraitC {}

fn foo<T>()
where
    for<const N: u8 = { T::A }> T: TraitA<AsA = impl TraitB<AsB = impl TraitC>>,
    //~^ ERROR late-bound const parameters cannot be used currently
    //~| ERROR defaults for generic parameters are not allowed in `for<...>` binders
    //~| ERROR `impl Trait` is not allowed in bounds
{
}

fn main() {}
