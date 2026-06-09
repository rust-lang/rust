//! Check that we correctly handle associated const bindings
//! where the RHS is a normalizable const projection (#151642).

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait { type const CT: bool; }

trait Bound { type const N: u32; }
impl Bound for () { type const N: u32 = 0; }

fn f() { let _: dyn Trait<CT = { <() as Bound>::N }>; }
//~^ ERROR the constant `0` is not of type `bool`
fn g(_: impl Trait<CT = { <() as Bound>::N }>) {}
//~^ ERROR the constant `0` is not of type `bool`

fn main() {}
