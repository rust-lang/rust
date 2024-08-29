//@ known-bug: rust-lang/rust#126646
mod foo {
    pub trait Callable {
        type Output;
        fn call() -> Self::Output;
    }

    impl<'a, V: ?Sized> Callable for &'a () {
        type Output = ();
    }
}
use foo::*;

fn test<'a>() -> impl Sized {
    <&'a () as Callable>::call()
}

fn main() {}
