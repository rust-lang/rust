#![feature(type_alias_impl_trait)]

type Foo<T> = impl Default;

#[allow(unused)]
fn foo<T: Default>(t: T) -> Foo<T> {
    t
    //~^ ERROR trait `Default` is not implemented for `T`
}

struct NotDefault;

fn main() {
    let _ = Foo::<NotDefault>::default();
}
