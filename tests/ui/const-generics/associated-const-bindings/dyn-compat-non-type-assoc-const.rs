// Ensure that traits with non-type associated consts require explicit dyn bindings.

//@ dont-require-annotations: NOTE

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait {
    const K: usize;
}

fn main() {
    let _: dyn Trait; //~ ERROR the value of the associated constant `K` in `Trait` must be specified

    // Specifying the non-type assoc const makes the dyn type fully constrained.
    let _: dyn Trait<K = 0>;
}
