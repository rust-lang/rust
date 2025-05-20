// Regression test for ICE from issue #140545
// The error message is confusing and wrong, but that's a different problem (#139350)
//@ edition:2018

trait Foo {}
fn a(x: impl Foo) -> impl Foo {
    if true { x } else { a(a(x)) }
    //~^ ERROR: expected generic type parameter, found `impl Foo` [E0792]
    //~| ERROR: type parameter `impl Foo` is part of concrete type but not used in parameter list for the `impl Trait` type alias
}

fn main(){}
