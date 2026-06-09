//! Check that we correctly handle associated const bindings
//! in `dyn Trait` where the RHS is a const param (#151642).

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait { type const CT: bool; }

fn f<const N: i32>() {
    let _: dyn Trait<CT = { N }>;
    //~^ ERROR the constant `N` is not of type `bool`
}
fn main() {}
