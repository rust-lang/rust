//@ revisions: current next
//@ ignore-compare-mode-next-solver (explicit revisions)
//@[next] compile-flags: -Znext-solver
//@[next] check-pass
#![feature(type_alias_impl_trait)]

type A = impl Foo;
type B = impl Foo;

trait Foo {}

fn muh(x: A) -> B {
    if false {
        return x;  // B's hidden type is A (opaquely)
        //[current]~^ ERROR opaque type's hidden type cannot be another opaque type
    }
    Bar // A's hidden type is `Bar`, because all the return types are compared with each other
}

struct Bar;
impl Foo for Bar {}

fn main() {}
