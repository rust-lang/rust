// Test that we can resolve type-relative associated const paths inside const parameter defaults
// where the self type is a simple type parameter.

//@ check-pass
#![feature(min_generic_const_args, macroless_generic_const_args)]

trait Trait {
    #[rustc_always_gca]
    const CT: usize;
}

// Below, `T::CT` resolves to `<T as Trait>::CT` since the owner has bound `T: Trait`.

struct Owner<T: Trait, const N: usize = { T::CT }>(T);

fn main() {}
