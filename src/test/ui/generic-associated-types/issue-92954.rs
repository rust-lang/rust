// check-pass

#![feature(generic_associated_types)]

pub trait Foo {
    type Assoc<'c>;
    fn function() -> for<'x> fn(Self::Assoc<'x>);
}

fn main() {}
