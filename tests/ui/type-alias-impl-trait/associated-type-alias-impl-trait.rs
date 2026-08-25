#![feature(type_alias_impl_trait)]
//@ build-pass (FIXME(62277): could be check-pass?)

trait Bar {}
struct Dummy;
impl Bar for Dummy {}

trait Foo {
    type Assoc: Bar;
    fn foo() -> Self::Assoc;
    fn bar() -> Self::Assoc;
}

type Helper = impl Bar;

impl Foo for i32 {
    type Assoc = Helper;
    #[define_opaque(Helper)]
    fn foo() -> Helper {
        Dummy
    }
    #[define_opaque(Helper)]
    fn bar() -> Helper {
        Dummy
    }
}

fn main() {}
