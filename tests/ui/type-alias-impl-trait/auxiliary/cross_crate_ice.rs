// Crate that exports an opaque `impl Trait` type. Used for testing cross-crate.

#![crate_type = "rlib"]
#![feature(type_alias_impl_trait)]

pub type Foo = impl std::fmt::Debug;

#[define_opaque(Foo)]
pub fn foo() -> Foo {
    5
}
