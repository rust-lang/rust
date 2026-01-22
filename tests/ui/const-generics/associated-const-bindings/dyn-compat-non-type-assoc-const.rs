// Ensure that traits with non-type associated consts are dyn *in*compatible.

//@ dont-require-annotations: NOTE

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    const K: usize;
    //~^ NOTE it contains associated const `K` that's not marked `#[type_const]`
}

fn main() {
    let _: dyn Trait; //~ ERROR the trait `Trait` is not dyn compatible

    // Check that specifying the non-type assoc const doesn't "magically make it work".
    let _: dyn Trait<K = 0>;
    //~^ ERROR the trait `Trait` is not dyn compatible
    //~| ERROR use of trait associated const without `#[type_const]`
}
