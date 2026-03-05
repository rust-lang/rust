#![feature(fn_delegation)]
#![allow(incomplete_features)]

fn main() {
    trait Foo {}
    fn foo<const N: dyn for<'a> Foo>() {}
    //~^ ERROR `(dyn Foo + 'static)` is forbidden as the type of a const generic parameter
    //~| ERROR `(dyn Foo + 'static)` is forbidden as the type of a const generic parameter
    reuse foo;
    //~^ ERROR the name `foo` is defined multiple times
}
