// Ensure that we reject the `Self` type parameter in supertrait bounds of dyn-compatible traits
// even if they're part of a "`Self` projection" (contrary to method signatures and the type of
// assoc consts).

//@ dont-require-annotations: NOTE

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait: SuperTrait<{ Self::N }> {
//~^ NOTE it uses `Self` as a type parameter
    #[type_const]
    const N: usize;
}

trait SuperTrait<const N: usize> {}

fn main() {
    let _: dyn Trait; //~ ERROR the trait `Trait` is not dyn compatible
}
