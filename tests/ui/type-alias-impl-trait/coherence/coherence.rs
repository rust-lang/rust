//@ aux-build:foreign-crate.rs
//@ revisions: classic next
//@[next] compile-flags: -Znext-solver
#![feature(type_alias_impl_trait)]

extern crate foreign_crate;

trait LocalTrait {}
impl<T> LocalTrait for foreign_crate::ForeignType<T> {}

type AliasOfForeignType<T> = impl LocalTrait;
#[define_opaque(AliasOfForeignType)]
fn use_alias<T>(val: T) -> AliasOfForeignType<T> {
    foreign_crate::ForeignType(val)
}

impl foreign_crate::ForeignTrait for AliasOfForeignType<()> {}
//~^ ERROR only traits defined in the current crate can be implemented for arbitrary types

fn main() {}
