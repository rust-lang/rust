// aux-build:foreign-crate.rs
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

extern crate foreign_crate;

trait LocalTrait {}
impl<T> LocalTrait for foreign_crate::ForeignType<T> {}

type AliasOfForeignType<T> = impl LocalTrait;
fn use_alias<T>(val: T) -> AliasOfForeignType<T> {
    foreign_crate::ForeignType(val)
}

impl<T> foreign_crate::ForeignTrait for AliasOfForeignType<T> {}
//~^ ERROR the type parameter `T` is not constrained by the impl trait, self type, or predicates

fn main() {}
