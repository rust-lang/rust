#![feature(type_alias_impl_trait)]

type Foo<T> = impl Default;

#[allow(unused)]
#[define_opaque(Foo)]
fn foo<T: Default>(t: T) -> Foo<T> {
    t
    //~^ ERROR: the trait bound `T: Default` is not satisfied
}

struct NotDefault;

fn main() {
    let _ = Foo::<NotDefault>::default();
}
