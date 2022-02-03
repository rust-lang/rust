#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;
type Bar = impl Trait<Foo>;

trait Trait<T> {}

impl<T, U> Trait<T> for U {}

fn bar() -> Bar {
    42
}

fn main() {
    println!("{:?}", bar());
    //~^ ERROR `impl Trait<Opaque(DefId(0:4 ~ nested[14f6]::Foo::{opaque#0}), [])>` doesn't implement `Debug`
}
