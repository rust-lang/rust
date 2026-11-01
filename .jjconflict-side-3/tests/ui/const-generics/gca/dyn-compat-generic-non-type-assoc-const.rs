// Ensure that traits with generic non-type associated consts are dyn *in*compatible,
// even when non-type associated const equality is enabled by `gca_const_items`.

//@ dont-require-annotations: NOTE
//@ compile-flags: -Znext-solver=globally

#![feature(gca_const_items, generic_const_items, gca_min_const_items)]
#![expect(incomplete_features)]

trait Trait {
    const ASSOC<const N: usize>: usize;
    //~^ NOTE it contains generic associated const `ASSOC`
}

fn main() {
    let _: dyn Trait;
    //~^ ERROR the trait `Trait` is not dyn compatible
}
