//@ compile-flags: -Znext-solver

// Check that we consider the reached depth of global cache
// entries when detecting overflow. We would otherwise be unstable
// wrt to incremental compilation.
#![recursion_limit = "9"]

trait Trait {}

struct Inc<T>(T);

impl<T: Trait> Trait for Inc<T> {}
impl Trait for () {}

fn impls_trait<T: Trait>() {}

type Four<T> = Inc<Inc<Inc<Inc<T>>>>;

fn main() {
    impls_trait::<Four<Four<()>>>();
    impls_trait::<Four<Four<Four<Four<()>>>>>();
    //~^ ERROR overflow evaluating the requirement
}
