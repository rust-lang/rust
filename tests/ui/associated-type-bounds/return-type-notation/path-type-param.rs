#![feature(return_type_notation)]

trait Foo {
    fn method<T>() -> impl Sized;
}

fn test<T: Foo>()
where
    <T as Foo>::method(..): Send,
    //~^ ERROR return type notation is not allowed for functions that have type parameters
{
}

fn test_type_dependent<T: Foo>()
where
    <T as Foo>::method(..): Send,
    //~^ ERROR return type notation is not allowed for functions that have type parameters
{
}

fn main() {}
