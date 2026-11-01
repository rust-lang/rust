// Ensure that traits with generic associated consts (GACs) are dyn *in*compatible.
// It would be very hard to make dyn Trait with GACs sound just like with GATs.

//@ dont-require-annotations: NOTE

#![feature(gca_min_const_items, generic_const_items)]
#![expect(incomplete_features)]

trait Trait {
    const POLY<T>: T;
    //~^ NOTE it contains generic associated const `POLY`
}

fn main() {
    let _: dyn Trait; //~ ERROR the trait `Trait` is not dyn compatible
}
