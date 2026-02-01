//@ aux-build:foreign-crate.rs
//@ revisions: current next
//@[next] compile-flags: -Znext-solver
//@ ignore-compare-mode-next-solver (explicit revisions)
#![feature(type_alias_impl_trait)]

// See also <https://github.com/rust-lang/rust/issues/130978>.
// FIXME(fmease): Add a more descriptive comment.

extern crate foreign_crate;

trait LocalTrait {}
impl<T> LocalTrait for foreign_crate::ForeignType<T> {}

type AliasOfForeignType<T> = impl LocalTrait;
#[define_opaque(AliasOfForeignType)]
fn use_alias<T>(val: T) -> AliasOfForeignType<T> {
    foreign_crate::ForeignType(val)
}

impl foreign_crate::ForeignTrait for AliasOfForeignType<()> {}
//[current]~^ ERROR opaque type `AliasOfForeignType<()>` must be used as the argument to some local type
//[next]~^^ ERROR type parameters and opaque types must be used as the argument to some local type

fn main() {}
