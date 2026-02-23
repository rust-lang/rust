// Traits with type associated consts are dyn compatible.
// Check that we allow the corresp. trait object types if all assoc consts are specified.

//@ check-pass

#![feature(min_generic_const_args)]
#![expect(incomplete_features)]

trait Trait: SuperTrait<C = 3> {
    type const K: usize;
}

trait SuperTrait {
    type const Q: usize;
    type const C: usize;
}

trait Bound {
    type const N: usize;
}

impl Bound for () {
    type const N: usize = 10;
}

fn main() {
    let _: dyn Trait<K = 1, Q = 2>;

    let obj: &dyn Bound<N = 10> = &();
    _ = identity(obj);

    fn identity(x: &(impl ?Sized + Bound<N = 10>)) -> &(impl ?Sized + Bound<N = 10>) { x }
}
