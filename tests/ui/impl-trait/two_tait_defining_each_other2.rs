//@ revisions: current next
//@[next] compile-flags: -Znext-solver
#![feature(type_alias_impl_trait)]

type A = impl Foo; //[current]~ ERROR unconstrained opaque type
type B = impl Foo;

trait Foo {}

fn muh(x: A) -> B {
    //[next]~^ ERROR type annotations needed: cannot satisfy `_ == A`
    x // B's hidden type is A (opaquely)
    //[current]~^ ERROR opaque type's hidden type cannot be another opaque type
}

struct Bar;
impl Foo for Bar {}

fn main() {}
