#![feature(associated_const_equality)]

trait Owner<K> {
    const C: K;
}
impl<K: ConstDefault> Owner<K> for () {
    const C: K = K::DEFAULT;
}

trait ConstDefault {
    const DEFAULT: Self;
}

fn user() -> impl Owner<dyn Sized, C = 0> {}
//~^ ERROR: the trait bound `(dyn Sized + 'static): ConstDefault` is not satisfied
//~| ERROR: the size for values of type `(dyn Sized + 'static)` cannot be known at compilation time
//~| ERROR: the trait `Sized` is not dyn compatible
//~| ERROR: mismatched types

fn main() {}
