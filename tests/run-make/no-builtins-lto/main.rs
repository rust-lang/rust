#![feature(no_core, start, lang_items)]
#![no_std]
// We use `no_core` to reduce the LTO products is small enough.
#![no_core]

extern crate no_builtins;
extern crate foo;

#[cfg_attr(unix, link(name = "c"))]
#[cfg_attr(target_env = "msvc", link(name = "msvcrt"))]
extern "C" {}

#[start]
fn main(_: isize, p: *const *const u8) -> isize {
    // Make sure the symbols are retained.
    unsafe { bar(*p as *mut u8, *p); }
    0
}

#[no_mangle]
#[inline(never)]
pub unsafe extern "C" fn bar(dest: *mut u8, src: *const u8) {
    no_builtins::no_builtins(dest, src);
    // should call `@llvm.memcpy`
    foo::memcpy(dest, src, 1024);
}

#[lang = "eh_personality"]
fn eh_personality() {}
