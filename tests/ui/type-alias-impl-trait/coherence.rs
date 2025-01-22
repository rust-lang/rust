//@ aux-build:foreign-crate.rs
//@ revisions: classic next
//@[next] compile-flags: -Znext-solver
#![feature(type_alias_impl_trait)]

// See also <https://github.com/rust-lang/rust/issues/130978>.

extern crate foreign_crate;

trait LocalTrait {}
impl<T> LocalTrait for foreign_crate::ForeignType<T> {}

type AliasOfForeignType<T> = impl LocalTrait;
fn use_alias<T>(val: T) -> AliasOfForeignType<T> {
    foreign_crate::ForeignType(val)
}

impl foreign_crate::ForeignTrait for AliasOfForeignType<()> {}
//~^ ERROR opaque type `AliasOfForeignType<()>` must be used as the argument to some local type

fn main() {}
