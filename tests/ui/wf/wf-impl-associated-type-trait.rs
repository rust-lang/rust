// Check that we require that associated types in an impl are well-formed.


#![allow(dead_code)]

pub trait MyHash { }

pub struct MySet<T:MyHash> {
    data: Vec<T>
}

pub trait Foo {
    type Bar;
}

impl<T> Foo for T {
    type Bar = MySet<T>;
    //~^ ERROR the trait bound `T: MyHash` is not satisfied
}


fn main() { }
