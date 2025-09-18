//@ add-core-stubs
//@ needs-llvm-components: x86
//@ compile-flags: --target=i686-pc-windows-gnu --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items, c_variadic)]

// Test that ABIs for which C-variadics are not supported report an error.

extern crate minicore;
use minicore::*;

#[rustfmt::skip]
mod foreign {
    extern "Rust"  { fn rust_foreign_explicit(_: ...); }
    //~^ ERROR C-variadic functions with the "Rust" calling convention are not supported
    extern "C"  { fn c_foreign(_: ...); }
    extern "C-unwind"  { fn c_unwind_foreign(_: ...); }
    extern "cdecl"  { fn cdecl_foreign(_: ...); }
    extern "cdecl-unwind"  { fn cdecl_unwind_foreign(_: ...); }
    extern "stdcall"  { fn stdcall_foreign(_: ...); }
    //~^ ERROR C-variadic functions with the "stdcall" calling convention are not supported
    extern "stdcall-unwind"  { fn stdcall_unwind_foreign(_: ...); }
    //~^ ERROR C-variadic functions with the "stdcall-unwind" calling convention are not supported
    extern "thiscall"  { fn thiscall_foreign(_: ...); }
    //~^ ERROR C-variadic functions with the "thiscall" calling convention are not supported
    extern "thiscall-unwind"  { fn thiscall_unwind_foreign(_: ...); }
    //~^ ERROR C-variadic functions with the "thiscall-unwind" calling convention are not supported
}

#[lang = "va_list"]
struct VaList(*mut u8);

unsafe fn rust_free(_: ...) {}
//~^ ERROR `...` is not supported for non-extern functions
unsafe extern "Rust" fn rust_free_explicit(_: ...) {}
//~^ ERROR `...` is not supported for `extern "Rust"` functions

unsafe extern "C" fn c_free(_: ...) {}
unsafe extern "C-unwind" fn c_unwind_free(_: ...) {}

unsafe extern "cdecl" fn cdecl_free(_: ...) {}
//~^ ERROR `...` is not supported for `extern "cdecl"` functions
unsafe extern "cdecl-unwind" fn cdecl_unwind_free(_: ...) {}
//~^ ERROR `...` is not supported for `extern "cdecl-unwind"` functions
unsafe extern "stdcall" fn stdcall_free(_: ...) {}
//~^ ERROR `...` is not supported for `extern "stdcall"` functions
unsafe extern "stdcall-unwind" fn stdcall_unwind_free(_: ...) {}
//~^ ERROR `...` is not supported for `extern "stdcall-unwind"` functions
unsafe extern "thiscall" fn thiscall_free(_: ...) {}
//~^ ERROR `...` is not supported for `extern "thiscall"` functions
unsafe extern "thiscall-unwind" fn thiscall_unwind_free(_: ...) {}
//~^ ERROR `...` is not supported for `extern "thiscall-unwind"` functions

struct S;

impl S {
    unsafe fn rust_method(_: ...) {}
    //~^ ERROR `...` is not supported for non-extern functions
    unsafe extern "Rust" fn rust_method_explicit(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "Rust"` functions

    unsafe extern "C" fn c_method(_: ...) {}
    unsafe extern "C-unwind" fn c_unwind_method(_: ...) {}

    unsafe extern "cdecl" fn cdecl_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "cdecl"` functions
    unsafe extern "cdecl-unwind" fn cdecl_unwind_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "cdecl-unwind"` functions
    unsafe extern "stdcall" fn stdcall_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "stdcall"` functions
    unsafe extern "stdcall-unwind" fn stdcall_unwind_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "stdcall-unwind"` functions
    unsafe extern "thiscall" fn thiscall_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "thiscall"` functions
    unsafe extern "thiscall-unwind" fn thiscall_unwind_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "thiscall-unwind"` functions
}

trait T {
    unsafe fn rust_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for non-extern functions
    unsafe extern "Rust" fn rust_trait_method_explicit(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "Rust"` functions

    unsafe extern "C" fn c_trait_method(_: ...) {}
    unsafe extern "C-unwind" fn c_unwind_trait_method(_: ...) {}

    unsafe extern "cdecl" fn cdecl_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "cdecl"` functions
    unsafe extern "cdecl-unwind" fn cdecl_unwind_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "cdecl-unwind"` functions
    unsafe extern "stdcall" fn stdcall_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "stdcall"` functions
    unsafe extern "stdcall-unwind" fn stdcall_unwind_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "stdcall-unwind"` functions
    unsafe extern "thiscall" fn thiscall_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "thiscall"` functions
    unsafe extern "thiscall-unwind" fn thiscall_unwind_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "thiscall-unwind"` functions
}

impl T for S {
    unsafe fn rust_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for non-extern functions
    unsafe extern "Rust" fn rust_trait_method_explicit(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "Rust"` functions

    unsafe extern "C" fn c_trait_method(_: ...) {}
    unsafe extern "C-unwind" fn c_unwind_trait_method(_: ...) {}

    unsafe extern "cdecl" fn cdecl_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "cdecl"` functions
    unsafe extern "cdecl-unwind" fn cdecl_unwind_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "cdecl-unwind"` functions
    unsafe extern "stdcall" fn stdcall_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "stdcall"` functions
    unsafe extern "stdcall-unwind" fn stdcall_unwind_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "stdcall-unwind"` functions
    unsafe extern "thiscall" fn thiscall_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "thiscall"` functions
    unsafe extern "thiscall-unwind" fn thiscall_unwind_trait_method(_: ...) {}
    //~^ ERROR `...` is not supported for `extern "thiscall-unwind"` functions
}
