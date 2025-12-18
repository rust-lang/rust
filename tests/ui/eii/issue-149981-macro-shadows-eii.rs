//@ check-pass
//@ compile-flags: --crate-type rlib
// Regression test for #149981: ICE when a macro shadows an EII function name.
// The compiler used to panic with "called Option::unwrap() on a None value"
// in rustc_metadata/src/eii.rs when collecting EII declarations.
//
// The issue occurred when a macro_rules! and an #[eii] function shared the
// same name. During EII collection, the compiler would try to look up the
// EiiExternTarget attribute on the macro DefId (which doesn't have it),
// causing an unwrap() on None to panic.

#![feature(extern_item_impls)]

// Define a macro with the same name as the EII declaration below.
// This shadows the generated EII macro.
macro_rules! foo {
    () => {};
}

// This EII declaration generates a macro also named `foo`.
#[eii]
fn foo();

// Attempting to use the EII would look up `foo` and find the macro_rules,
// which doesn't have the EiiExternTarget attribute - this used to ICE.
#[foo]
fn foo_impl() {}
