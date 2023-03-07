#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

pub struct Foo<T, const H: T>(T)
//~^ ERROR the type of const parameters must not depend on other generic parameters
where
    [(); 1]:;

fn main() {}
