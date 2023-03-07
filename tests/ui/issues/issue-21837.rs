pub trait Bound {}
pub struct Foo<T: Bound>(T);

pub trait Trait1 {}
impl<T: Bound> Trait1 for Foo<T> {}

pub trait Trait2 {}
impl<T> Trait2 for Foo<T> {} //~ ERROR the trait bound `T: Bound` is not satisfied

fn main() {}
