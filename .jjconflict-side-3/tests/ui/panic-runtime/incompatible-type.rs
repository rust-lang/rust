// Check that rust_eh_personality can have a different type signature than the
// one hardcoded in the compiler.  Regression test for #70117. Used to fail with:
//
// Assertion `isa<X>(Val) && "cast<Ty>() argument of incompatible type!"' failed.
//
//@ build-pass
//@ compile-flags: --crate-type=lib -Ccodegen-units=1
#![no_std]
#![panic_runtime]
#![feature(panic_runtime)]
#![feature(rustc_attrs)]

pub struct DropMe;

impl Drop for DropMe {
    fn drop(&mut self) {}
}

pub fn test(_: DropMe) {
    unreachable!();
}

#[rustc_std_internal_symbol]
pub unsafe extern "C" fn rust_eh_personality(
    _version: i32,
    _actions: i32,
    _exception_class: u64,
    _exception_object: *mut (),
    _context: *mut (),
) -> i32 {
    loop {}
}
