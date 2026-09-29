//@ build-pass
//@ compile-flags: --crate-type=dylib
//@ needs-dynamic-linking
//@ needs-crate-type: dylib

#![allow(trivial_bounds)]

pub fn foo()
where
    for<'a> [u8]: Sized,
{
}
