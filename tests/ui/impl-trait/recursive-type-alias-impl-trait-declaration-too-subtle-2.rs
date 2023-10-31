#![feature(type_alias_impl_trait)]

// check-pass

type Foo = impl PartialEq<(Foo, i32)>;

struct Bar;

impl PartialEq<(Foo, i32)> for Bar {
    fn eq(&self, _other: &(Foo, i32)) -> bool {
        true
    }
}

fn foo() -> Foo {
    Bar
}

fn main() {}
