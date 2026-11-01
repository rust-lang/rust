#![feature(type_alias_impl_trait)]

pub trait Bar<T> {
    type Item;
}

type Foo = impl Bar<Foo, Item = Foo>;
#[define_opaque(Foo)]
fn crash(x: Foo) -> Foo {
    //~^ ERROR overflow evaluating the requirement `<Foo as Bar<Foo>>::Item == Foo`
    x
}

fn main() {}
