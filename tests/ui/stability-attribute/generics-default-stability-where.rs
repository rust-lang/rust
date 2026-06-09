//@ aux-build:unstable_generic_param.rs

extern crate unstable_generic_param;

use unstable_generic_param::*;

impl<T> Trait3<usize> for T where T: Trait2<usize> { //~ ERROR use of unstable library feature `unstable_default`
//~^ ERROR `T` must be used as the type parameter for some local type
    fn foo() -> usize { T::foo() }
}

fn main() {}
