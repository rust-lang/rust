//! This test used to ICE when trying to resolve the method call in the `test` function.

mod foo {
    pub trait Callable {
        type Output;
        fn call() -> Self::Output;
    }

    impl<'a, V: ?Sized> Callable for &'a () {
        //~^ ERROR: `V` is not constrained
        type Output = ();
    }
}
use foo::*;

fn test<'a>() -> impl Sized {
    <&'a () as Callable>::call()
}

fn main() {}
