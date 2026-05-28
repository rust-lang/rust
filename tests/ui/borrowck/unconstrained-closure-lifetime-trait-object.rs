// Regression test for #139004
use std::any::Any;

type B = Box<dyn for<'a> Fn(&(dyn Any + 'a)) -> Box<dyn Any + 'a>>;

fn foo<E>() -> B {
    Box::new(|e| Box::new(e.is::<E>()))
    //~^ ERROR the parameter type `E` may not live long enough
}

fn main() {}
