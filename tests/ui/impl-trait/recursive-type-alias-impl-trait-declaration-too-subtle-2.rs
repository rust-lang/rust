#![feature(type_alias_impl_trait)]

//@ check-pass

mod foo {
    pub type Foo = impl PartialEq<(Foo, i32)>;

    fn foo() -> Foo {
        super::Bar
    }
}
use foo::Foo;

struct Bar;

impl PartialEq<(Foo, i32)> for Bar {
    fn eq(&self, _other: &(Foo, i32)) -> bool {
        true
    }
}

fn main() {}
