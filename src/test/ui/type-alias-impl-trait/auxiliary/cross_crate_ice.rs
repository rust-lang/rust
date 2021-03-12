// Crate that exports an opaque `impl Trait` type. Used for testing cross-crate.

#![crate_type = "rlib"]
// revisions: min_tait full_tait
#![feature(min_type_alias_impl_trait)]
#![cfg_attr(full_tait, feature(type_alias_impl_trait))]
//[full_tait]~^ WARN incomplete

pub type Foo = impl std::fmt::Debug;

pub fn foo() -> Foo {
    5
}
