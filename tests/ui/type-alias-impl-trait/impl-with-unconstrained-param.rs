// Ensure that we don't ICE if associated type impl trait is used in an impl
// with an unconstrained type parameter.

trait X {
    type I;
    fn f() -> Self::I;
}

impl<T> X for () {
    //~^ ERROR the type parameter `T` is not constrained
    type I = impl Sized;
    fn f() -> Self::I {}
}

fn main() {}
