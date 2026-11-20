// Ensure that coroutine drop glue is valid when mixing different panic
// strategies. Regression test for #116953.
//
//@ no-prefer-dynamic
//@ build-pass
//@ aux-build:unwind-aux.rs
//@ compile-flags: -Cpanic=abort
//@ needs-unwind
//@ ignore-backends: gcc
extern crate unwind_aux;

pub fn main() {
    unwind_aux::run(String::new());
}
