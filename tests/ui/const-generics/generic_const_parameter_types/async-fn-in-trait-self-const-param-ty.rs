//@ edition: 2021
#![feature(generic_const_parameter_types)]
trait Foo {
    async fn foo<const N: Self>(); //~ ERROR
}

impl Foo for () {
    async fn foo<const N: Self>() {} //~ ERROR
}

fn main() {}
