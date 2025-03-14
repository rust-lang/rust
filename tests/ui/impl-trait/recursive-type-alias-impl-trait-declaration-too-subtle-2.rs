#![feature(type_alias_impl_trait)]

//@ check-pass

pub type Foo = impl PartialEq<(Foo, i32)>;

#[define_opaque(Foo)]
fn foo() -> Foo {
    Bar
}

struct Bar;

impl PartialEq<(Foo, i32)> for Bar {
    fn eq(&self, _other: &(Foo, i32)) -> bool {
        true
    }
}

fn main() {}
