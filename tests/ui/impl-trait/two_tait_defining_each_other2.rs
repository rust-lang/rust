#![feature(type_alias_impl_trait)]

type A = impl Foo; //~ ERROR unconstrained opaque type
type B = impl Foo;

trait Foo {}

fn muh(x: A) -> B {
    x // B's hidden type is A (opaquely)
    //~^ ERROR opaque type's hidden type cannot be another opaque type
}

struct Bar;
impl Foo for Bar {}

fn main() {}
