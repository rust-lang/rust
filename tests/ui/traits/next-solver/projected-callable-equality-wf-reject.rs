//@ check-fail
//@ compile-flags: -Znext-solver=globally

#![allow(dead_code)]

trait Lending {
    type View<'a> where Self: 'a;
}

fn missing_lifetime<'a, T>()
where
    T: Lending<View<'a> = ()>,
    //~^ ERROR the parameter type `T` may not live long enough
{
}

trait Required {}
trait Conditional {
    type View<'a> where &'a (): Required;
}

fn missing_trait<'a, T>()
where
    T: Conditional<View<'a> = ()>,
    //~^ ERROR the trait bound `&'a (): Required` is not satisfied
{
}

fn main() {}
