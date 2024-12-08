//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
#![feature(type_alias_impl_trait)]

type A = impl Foo;
type B = impl Foo;

trait Foo {}

fn muh(x: A) -> B {
    //[current]~^ ERROR: item does not constrain `A::{opaque#0}`
    //[next]~^^ ERROR: cannot satisfy `_ == A`
    x // B's hidden type is A (opaquely)
    //[current]~^ ERROR opaque type's hidden type cannot be another opaque type
}

struct Bar;
impl Foo for Bar {}

fn main() {}
