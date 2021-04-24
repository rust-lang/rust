// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

fn main() {}

type WrongGeneric<T: 'static> = impl 'static;
//~^ ERROR: at least one trait must be specified

fn wrong_generic<U: 'static, V: 'static>(_: U, v: V) -> WrongGeneric<U> {
//~^ ERROR type parameter `V` is part of concrete type but not used in parameter list
    v
}
