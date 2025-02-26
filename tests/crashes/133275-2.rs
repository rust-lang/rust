//@ known-bug: #133275
#![feature(const_trait_impl)]
#[const_trait]
pub trait Owo<X = <IntEnum as Uwu>::T> {}

#[const_trait]
trait Foo3<T>
where
    Self::Bar: Clone,
    Self::Baz: Clone,
{
    type Bar = Vec<Self::Baz>;
    type Baz = T;
    //~^ ERROR the trait bound `T: Clone` is not satisfied
}
