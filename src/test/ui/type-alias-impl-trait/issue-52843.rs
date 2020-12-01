#![feature(type_alias_impl_trait)]

type Foo<T> = impl Default;
//~^ ERROR: the trait bound `T: Default` is not satisfied

#[allow(unused)]
fn foo<T: Default>(t: T) -> Foo<T> {
    t
}

struct NotDefault;

fn main() {
    let _ = Foo::<NotDefault>::default();
}
