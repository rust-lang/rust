//@ build-pass
//@ compile-flags: --crate-type=dylib
//@ needs-dynamic-linking
//@ needs-crate-type: dylib

#![allow(trivial_bounds)]

// Regression test for #128682.
// A function with an impossible where-clause must not be exported from a dylib,
// since codegen does not emit a definition for it.
pub fn foo()
where
    for<'a> [u8]: Sized,
{
}
