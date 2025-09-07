//@ check-pass

#![deny(improper_ctypes)]

// Issue: https://github.com/rust-lang/rust/issues/66202
// This test checks that return types are normalized before being checked for FFI-safety, and that
// transparent newtype wrappers are FFI-safe if the type being wrapped is FFI-safe.

#[repr(transparent)]
pub struct W<T>(T);

extern "C" {
    pub fn bare() -> ();
    pub fn normalize() -> <() as ToOwned>::Owned;
    pub fn transparent() -> W<()>;
}

fn main() {}
