//@ only-wasm32
//@ aux-build:link-name-in-foreign-crate.rs
//@ compile-flags: --crate-type cdylib
//@ build-pass
//@ no-prefer-dynamic

extern crate link_name_in_foreign_crate;

// This test that the definition of a function named `close`, which collides
// with the `close` function in libc in theory, is handled correctly in
// cross-crate situations. The `link_name_in_foreign_crate` dependency declares
// `close` from a non-`env` wasm import module and then this crate attempts to
// use the symbol. This should properly ensure that the wasm module name is
// tagged as `test` and the `close` symbol, to LLD, is mangled, to avoid
// colliding with the `close` symbol in libc itself.

#[unsafe(no_mangle)]
pub extern "C" fn foo() {
    unsafe {
        link_name_in_foreign_crate::close(1);
    }
}
