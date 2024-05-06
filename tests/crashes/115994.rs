//@ known-bug: #115994
//@ compile-flags: -Cdebuginfo=2 --crate-type lib

// To prevent "overflow while adding drop-check rules".
use std::mem::ManuallyDrop;

pub enum Foo<U> {
    Leaf(U),

    Branch(BoxedFoo<BoxedFoo<U>>),
}

pub type BoxedFoo<U> = ManuallyDrop<Box<Foo<U>>>;

pub fn test() -> Foo<usize> {
    todo!()
}
