//@ edition: 2021

#![feature(return_type_notation)]

trait Foo {
    async fn bar<T>() {}

    async fn baz<const N: usize>() {}
}

fn test<T>()
where
    T: Foo<bar(..): Send, baz(..): Send>,
    //~^ ERROR return type notation is not allowed for functions that have const parameters
    //~| ERROR return type notation is not allowed for functions that have type parameters
{
}

fn main() {}
