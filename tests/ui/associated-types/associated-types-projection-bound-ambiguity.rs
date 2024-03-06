// Check that if we have multiple applicable projection bounds we pick one (for
// backwards compatibility reasons).

//@ check-pass
use std::ops::Mul;

trait A {
    type V;
    type U: Mul<Self::V, Output = ()> + Mul<(), Output = ()>;
}

fn g<T: A<V = ()>>() {
    let y: <T::U as Mul<()>>::Output = ();
}

fn main() {}
