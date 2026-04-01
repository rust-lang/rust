//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass

#![feature(type_alias_impl_trait)]

type A = impl Foo;
type B = impl Foo;

trait Foo {}

#[define_opaque(A, B)]
fn muh(x: A) -> B {
    if false {
        return Bar; // B's hidden type is Bar
    }
    x // A's hidden type is `Bar`, because all the hidden types of `B` are compared with each other
    //[current]~^ ERROR opaque type's hidden type cannot be another opaque type
}

struct Bar;
impl Foo for Bar {}

fn main() {}
