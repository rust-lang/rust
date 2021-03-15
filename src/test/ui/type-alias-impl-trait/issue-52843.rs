// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

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
