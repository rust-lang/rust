#![feature(type_alias_impl_trait)]

type Foo = impl std::fmt::Debug;
type Bar = impl Trait<Foo>;

trait Trait<T> {}

impl<T, U> Trait<T> for U {}

#[define_opaque(Bar)]
fn bar() -> Bar {
    //~^ ERROR: item does not constrain
    42
}

fn main() {
    println!("{:?}", bar());
    //~^ ERROR `Bar` doesn't implement `Debug`
}
