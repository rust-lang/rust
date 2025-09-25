// Regression test for ICE from issue #140545
// The error message is confusing and wrong, but that's a different problem (#139350)

//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] check-pass

trait Foo {}
fn a<T: Foo>(x: T) -> impl Foo {
    if true { x } else { a(a(x)) }
    //[current]~^ ERROR: expected generic type parameter, found `impl Foo` [E0792]
    //[current]~| ERROR: type parameter `T` is part of concrete type but not used in parameter list for the `impl Trait` type alias
}

fn main(){}
