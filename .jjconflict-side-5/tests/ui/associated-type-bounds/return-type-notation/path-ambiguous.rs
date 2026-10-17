#![feature(return_type_notation)]

trait A {
    fn method() -> impl Sized;
}
trait B {
    fn method() -> impl Sized;
}

fn ambiguous<T: A + B>()
where
    T::method(..): Send,
    //~^ ERROR ambiguous associated function `method` in bounds of `T`
{
}

trait Sub: A + B {}

fn ambiguous_via_supertrait<T: Sub>()
where
    T::method(..): Send,
    //~^ ERROR ambiguous associated function `method` in bounds of `T`
{
}

fn main() {}
