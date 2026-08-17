// Ensure that traits with generic non-type associated consts are dyn *in*compatible,
// even when non-type associated const equality is enabled by `generic_const_args`.

//@ dont-require-annotations: NOTE
//@ compile-flags: -Znext-solver=globally

#![feature(generic_const_args, generic_const_items, min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    const ASSOC<const N: usize>: usize;
    //~^ NOTE it contains generic associated const `ASSOC`
}

fn main() {
    let _: dyn Trait;
    //~^ ERROR the trait `Trait` is not dyn compatible
}
