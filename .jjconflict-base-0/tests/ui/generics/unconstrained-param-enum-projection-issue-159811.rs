//! Regression test for https://github.com/rust-lang/rust/issues/159811.

trait Child {
    type Error;
}

trait Site {
    type Child<'a>: Child
    where
        Self: 'a;
    type Error;
}

enum MyError<P> {
    Own,
    Parent(P),
}

impl<'a, S: Site> From<<S::Child<'a> as Child>::Error> for MyError<S::Error> {
    //~^ ERROR conflicting implementations of trait
    //~| ERROR the type parameter `S` is not constrained
    fn from(_: <S::Child<'a> as Child>::Error) -> Self {
        Self::Own
    }
}

fn main() {}
