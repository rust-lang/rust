#![feature(lang_items, no_core)]
#![no_std]
#![no_core]
#![crate_type = "lib"]
#![no_builtins]

extern crate foo;

#[no_mangle]
pub unsafe fn no_builtins(dest: *mut u8, src: *const u8) {
    // There should be no "undefined reference to `foo::foo'".
    foo::foo(dest, src);
    // should call `@memcpy` instead of `@llvm.memcpy`.
    foo::memcpy(dest, src, 1024);
}
