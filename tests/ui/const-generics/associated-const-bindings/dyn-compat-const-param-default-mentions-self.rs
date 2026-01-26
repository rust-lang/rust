// Test that we force users to explicitly specify const arguments for const parameters that
// have defaults if the default mentions the `Self` type parameter.

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait X<const N: usize = { <Self as Y>::N }> {}

trait Y {
    #[type_const]
    const N: usize;
}

impl<T: ?Sized> Y for T {
    #[type_const]
    const N: usize = 1;
}

fn main() {
    let _: dyn X; //~ ERROR the const parameter `N` must be explicitly specified
}
