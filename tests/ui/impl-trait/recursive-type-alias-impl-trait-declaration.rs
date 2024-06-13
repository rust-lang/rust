#![feature(type_alias_impl_trait)]

type Foo = impl PartialEq<(Foo, i32)>;

struct Bar;

impl PartialEq<(Bar, i32)> for Bar {
    fn eq(&self, _other: &(Bar, i32)) -> bool {
        true
    }
}

fn foo() -> Foo {
    //~^ ERROR can't compare `Bar` with `(Foo, i32)`
    Bar
}

fn main() {}
