#![feature(return_type_notation)]

trait A {
    #[allow(non_camel_case_types)]
    type bad;
}

fn fully_qualified<T: A>()
where
    <T as A>::method(..): Send,
    //~^ ERROR cannot find method or associated constant `method` in trait `A`
    <T as A>::bad(..): Send,
    //~^ ERROR expected method or associated constant, found associated type `A::bad`
{
}

fn type_dependent<T: A>()
where
    T::method(..): Send,
    //~^ ERROR associated function `method` not found for `T`
{
}

fn main() {}
