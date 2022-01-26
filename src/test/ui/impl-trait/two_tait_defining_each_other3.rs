#![feature(type_alias_impl_trait)]

// check-pass

type A = impl Foo;
type B = impl Foo;

trait Foo {}

fn muh(x: A) -> B {
    if false {
        return x;  // B's hidden type is A (opaquely)
    }
    Bar // A's hidden type is `Bar`, because all the return types are compared with each other
}

struct Bar;
impl Foo for Bar {}

fn main() {}
