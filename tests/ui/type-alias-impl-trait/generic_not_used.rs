#![feature(type_alias_impl_trait)]

fn main() {}

type WrongGeneric<T: 'static> = impl 'static;
//~^ ERROR: at least one trait must be specified

#[define_opaque(WrongGeneric)]
fn wrong_generic<U: 'static, V: 'static>(_: U, v: V) -> WrongGeneric<U> {
    //~^ ERROR type parameter `V` is part of concrete type but not used in parameter list
    v
}
