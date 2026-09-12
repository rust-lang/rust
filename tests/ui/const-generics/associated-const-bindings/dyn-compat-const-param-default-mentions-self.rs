// Test that we force users to explicitly specify const arguments for const parameters that
// have defaults if the default mentions the `Self` type parameter.

#![feature(min_generic_const_args, macroless_generic_const_args)]
#![expect(incomplete_features)]

trait X<const N: usize = { <Self as Y>::N }> {}

trait Y {
    #[rustc_always_gca]
    const N: usize;
}

impl<T: ?Sized> Y for T {
    const N: usize = core::direct_const_arg!(1);
}

fn main() {
    let _: dyn X; //~ ERROR the const parameter `N` must be explicitly specified
}
