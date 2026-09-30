// Test that we force users to explicitly specify const arguments for const parameters that
// have defaults if the default mentions the `Self` type parameter.

#![feature(gca_min_const_items, gca_macroless_args)]
#![expect(incomplete_features)]

use std::gca;

trait X<const N: usize = { <Self as Y>::N }> {}

trait Y {
    #[rustc_always_gca]
    const N: usize;
}

impl<T: ?Sized> Y for T {
    const N: usize = gca!(1);
}

fn main() {
    let _: dyn X; //~ ERROR the const parameter `N` must be explicitly specified
}
