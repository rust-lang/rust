//@ check-fail

pub trait Trait {
    type Assoc<'a> where Self: 'a;
}

pub trait Foo<T: Trait>
where
    for<'a> T::Assoc<'a>: Clone
{}

pub struct Type;

impl<T: Trait> Foo<T> for Type //~ ERROR: the parameter type `T` may not live long enough
where
    for<'a> T::Assoc<'a>: Clone
{}

fn main() {}
