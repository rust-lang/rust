#![allow(dead_code)]
//@ run-rustfix
// Check projection of an associated type out of a higher-ranked trait-bound
// in the context of a method definition in a trait.

pub trait Foo<T> {
    type A;

    fn get(&self, t: T) -> Self::A;
}

trait SomeTrait<I : for<'x> Foo<&'x isize>> {
    fn some_method(&self, arg: I::A);
    //~^ ERROR cannot use the associated type of a trait with uninferred generic parameters
}

trait AnotherTrait<I : for<'x> Foo<&'x isize>> {
    fn some_method(&self, arg: <I as Foo<&isize>>::A);
}

trait YetAnotherTrait<I : for<'x> Foo<&'x isize>> {
    fn some_method<'a>(&self, arg: <I as Foo<&'a isize>>::A);
}

trait Banana<'a> {
    type Assoc: Default;
}

struct Peach<X>(std::marker::PhantomData<X>);

impl<X: for<'a> Banana<'a>> Peach<X> {
    fn mango(&self) -> X::Assoc {
    //~^ ERROR cannot use the associated type of a trait with uninferred generic parameters
        Default::default()
    }
}

pub fn main() {}
