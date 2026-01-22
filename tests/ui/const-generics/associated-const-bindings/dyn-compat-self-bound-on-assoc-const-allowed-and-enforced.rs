// Ensure that the where-clause of assoc consts in dyn-compatible traits are allowed to freely
// reference the `Self` type parameter (contrary to methods) and that such where clauses are
// actually enforced.

#![feature(min_generic_const_args, generic_const_items)]
#![expect(incomplete_features)]

trait Trait {
    #[type_const]
    const N: i32 where Self: Bound;
}

impl Trait for () {
    #[type_const]
    const N: i32 = 0;
}

trait Bound {}

fn main() {
    let _: dyn Trait<N = 0>; // OK

    let _: &dyn Trait<N = 0> = &(); //~ ERROR the trait bound `(): Bound` is not satisfied
}
