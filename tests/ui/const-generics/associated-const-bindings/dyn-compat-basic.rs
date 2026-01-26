// Traits with type associated consts are dyn compatible.
// Check that we allow the corresp. trait object types if all assoc consts are specified.

//@ check-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait: SuperTrait<C = 3> {
    #[type_const]
    const K: usize;
}

trait SuperTrait {
    #[type_const]
    const Q: usize;
    #[type_const]
    const C: usize;
}

trait Bound {
    #[type_const]
    const N: usize;
}

impl Bound for () {
    #[type_const]
    const N: usize = 10;
}

fn main() {
    let _: dyn Trait<K = 1, Q = 2>;

    let obj: &dyn Bound<N = 10> = &();
    _ = identity(obj);

    fn identity(x: &(impl ?Sized + Bound<N = 10>)) -> &(impl ?Sized + Bound<N = 10>) { x }
}
