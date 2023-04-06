#![feature(type_alias_impl_trait)]

type Foo = impl PartialEq<(Foo, i32)>;

struct Bar;

impl PartialEq<(Foo, i32)> for Bar {
    fn eq(&self, _other: &(Foo, i32)) -> bool {
        true
    }
}

#[defines(Foo)]
fn foo() -> Foo {
    //~^ ERROR can't compare `Bar` with `(Bar, i32)`
    Bar
}

fn main() {}
