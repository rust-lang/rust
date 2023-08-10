#![feature(lang_items, no_core)]
#![no_std]
#![no_core]
#![crate_type = "lib"]

#[inline(never)]
#[no_mangle]
pub unsafe fn foo(dest: *mut u8, src: *const u8) {
    // should call `@llvm.memcpy`.
    memcpy(dest, src, 1024);
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn memcpy(dest: *mut u8, src: *const u8, _n: usize) -> *mut u8 {
    *dest = 0;
    return src as *mut u8;
}

#[lang = "sized"]
trait Sized {}
#[lang = "copy"]
trait Copy {}
impl Copy for *mut u8 {}
impl Copy for *const u8 {}

#[lang = "drop_in_place"]
#[allow(unconditional_recursion)]
pub unsafe fn drop_in_place<T: ?Sized>(to_drop: *mut T) {
    // Code here does not matter - this is replaced by the
    // real drop glue by the compiler.
    drop_in_place(to_drop);
}
