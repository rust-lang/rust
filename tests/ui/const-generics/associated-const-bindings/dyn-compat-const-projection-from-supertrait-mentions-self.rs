// Test that we force users to explicitly specify associated constants (via bindings)
// which reference the `Self` type parameter.

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait X: Y<K = { Self::Q }> {
    #[type_const]
    const Q: usize;
}

trait Y {
    #[type_const]
    const K: usize;
}

fn main() {
    let _: dyn X<Q = 10>;
    //~^ ERROR the value of the associated constant `K` in `Y` must be specified
}
