#![feature(return_type_notation)]
//~^ WARN the feature `return_type_notation` is incomplete

trait Tr {
    const CONST: usize;

    fn method() -> impl Sized;
}

fn foo<T: Tr>()
where
    T::method(..): Send,
    //~^ ERROR return type notation not allowed in this position yet
    //~| ERROR expected type, found function
    <T as Tr>::method(..): Send,
    //~^ ERROR return type notation not allowed in this position yet
    //~| ERROR expected associated type, found associated function `Tr::method`
{
    let _ = T::CONST::(..);
    //~^ ERROR return type notation not allowed in this position yet
    let _: T::method(..);
    //~^ ERROR return type notation not allowed in this position yet
    //~| ERROR expected type, found function
}

fn main() {}
