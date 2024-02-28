pub trait Bound {}
pub struct Foo<T: Bound>(T);

pub trait Trait1 {}
impl<T: Bound> Trait1 for Foo<T> {}

pub trait Trait2 {}
impl<T> Trait2 for Foo<T> {} //~ ERROR trait `Bound` is not implemented for `T`

fn main() {}
