#![feature(existential_type)]
// compile-pass

trait Bar {}
struct Dummy;
impl Bar for Dummy {}

trait Foo {
    type Assoc: Bar;
    fn foo() -> Self::Assoc;
    fn bar() -> Self::Assoc;
}

existential type Helper: Bar;

impl Foo for i32 {
    type Assoc = Helper;
    fn foo() -> Helper {
        Dummy
    }
    fn bar() -> Helper {
        Dummy
    }
}

fn main() {}
