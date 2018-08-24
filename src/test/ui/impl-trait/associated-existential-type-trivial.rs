#![feature(existential_type)]
// compile-pass

trait Bar {}
struct Dummy;
impl Bar for Dummy {}

trait Foo {
    type Assoc: Bar;
    fn foo() -> Self::Assoc;
}

impl Foo for i32 {
    existential type Assoc: Bar;
    fn foo() -> Self::Assoc {
        Dummy
    }
}

fn main() {}
