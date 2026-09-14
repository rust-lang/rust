// Traits with type associated consts are dyn compatible.
// Check that we allow the corresp. trait object types if all assoc consts are specified.

//@ check-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait: SuperTrait<C = 3> {
    #[rustc_always_gca]
    const K: usize;
}

trait SuperTrait {
    #[rustc_always_gca]
    const Q: usize;
    #[rustc_always_gca]
    const C: usize;
}

trait Bound {
    #[rustc_always_gca]
    const N: usize;
}

impl Bound for () {
    const N: usize = core::direct_const_arg!(10);
}

fn main() {
    let _: dyn Trait<K = 1, Q = 2>;

    let obj: &dyn Bound<N = 10> = &();
    _ = identity(obj);

    fn identity(x: &(impl ?Sized + Bound<N = 10>)) -> &(impl ?Sized + Bound<N = 10>) {
        x
    }
}
