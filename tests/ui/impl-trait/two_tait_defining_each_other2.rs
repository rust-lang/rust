// revisions: current next
//[next] compile-flags: -Ztrait-solver=next
#![feature(type_alias_impl_trait)]

type A = impl Foo; //[current]~ ERROR unconstrained opaque type
type B = impl Foo;

trait Foo {}

fn muh(x: A) -> B {
    x // B's hidden type is A (opaquely)
    //[current]~^ ERROR opaque type's hidden type cannot be another opaque type
    //[next]~^^ ERROR type annotations needed: cannot satisfy `A <: B`
}

struct Bar;
impl Foo for Bar {}

fn main() {}
