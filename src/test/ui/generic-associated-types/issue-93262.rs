// check-pass

#![feature(generic_associated_types)]

pub trait Trait {
    type Assoc<'a> where Self: 'a;
}

pub trait Foo<T: Trait>
where
    for<'a> T::Assoc<'a>: Clone
{}

pub struct Type;

impl<T: Trait> Foo<T> for Type
where
    for<'a> T::Assoc<'a>: Clone
{}

fn main() {}
