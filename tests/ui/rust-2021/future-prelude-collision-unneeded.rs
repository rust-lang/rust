//@ edition:2018
//@ check-pass
#![allow(unused)]
#![deny(rust_2021_prelude_collisions)]

struct S;

impl S {
    fn try_into(self) -> S {
        S
    }
}

struct X;

trait Hey {
    fn from_iter(_: i32) -> Self;
}

impl Hey for X {
    fn from_iter(_: i32) -> Self {
        X
    }
}

struct Y<T>(T);

impl Hey for Y<i32> {
    fn from_iter(_: i32) -> Self {
        Y(0)
    }
}

struct Z<T>(T);

impl Hey for Z<i32> {
    fn from_iter(_: i32) -> Self {
        Z(0)
    }
}

impl std::iter::FromIterator<u32> for Z<u32> {
    fn from_iter<T: IntoIterator<Item = u32>>(_: T) -> Self {
        todo!()
    }
}

fn main() {
    // See https://github.com/rust-lang/rust/issues/86633
    let s = S;
    let s2 = s.try_into();

    // Check that we do not issue suggestions for types that do not implement `FromIter`.
    //
    // See https://github.com/rust-lang/rust/issues/86902
    X::from_iter(1);
    Y::from_iter(1);
    Y::<i32>::from_iter(1);
    Z::<i32>::from_iter(1);
}
