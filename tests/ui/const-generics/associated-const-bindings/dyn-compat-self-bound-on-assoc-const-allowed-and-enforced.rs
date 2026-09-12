// Ensure that the where-clause of assoc consts in dyn-compatible traits are allowed to freely
// reference the `Self` type parameter (contrary to methods) and that such where clauses are
// actually enforced.

#![feature(min_generic_const_args, generic_const_items)]
#![expect(incomplete_features)]

trait Trait {
    #[rustc_always_gca]
    const N: i32 where Self: Bound;
}

impl Trait for () {
    const N: i32 = core::direct_const_arg!(0);
}

trait Bound {}

fn main() {
    let _: dyn Trait<N = 0>; // OK

    let _: &dyn Trait<N = 0> = &(); //~ ERROR the trait bound `(): Bound` is not satisfied
}
