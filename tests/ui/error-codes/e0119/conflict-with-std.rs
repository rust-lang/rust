use std::marker::PhantomData;
use std::convert::{TryFrom, AsRef};

struct Q;
impl AsRef<Q> for Box<Q> { //~ ERROR conflicting implementations
    fn as_ref(&self) -> &Q {
        &**self
    }
}

struct S;
impl From<S> for S { //~ ERROR conflicting implementations
    fn from(s: S) -> S {
        s
    }
}

struct X;
impl TryFrom<X> for X { //~ ERROR conflicting implementations
    type Error = ();
    fn try_from(u: X) -> Result<X, ()> {
        Ok(u)
    }
}

fn main() {}
