#![feature(return_type_notation)]

trait A<'a> {
    fn method() -> impl Sized;
}
trait B: for<'a> A<'a> {}

fn higher_ranked<T>()
where
    T: for<'a> A<'a>,
    T::method(..): Send,
    //~^ ERROR cannot use the associated function of a trait with uninferred generic parameters
{
}

fn higher_ranked_via_supertrait<T>()
where
    T: B,
    T::method(..): Send,
    //~^ ERROR cannot use the associated function of a trait with uninferred generic parameters
{
}

fn main() {}
