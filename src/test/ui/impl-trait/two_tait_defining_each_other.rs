#![feature(type_alias_impl_trait)]

// check-pass

type A = impl Foo;
type B = impl Foo;

trait Foo {}

fn muh(x: A) -> B {
    if false {
        return Bar; // B's hidden type is Bar
    }
    x // A's hidden type is `Bar`, because all the hidden types of `B` are compared with each other
}

struct Bar;
impl Foo for Bar {}

fn main() {}
