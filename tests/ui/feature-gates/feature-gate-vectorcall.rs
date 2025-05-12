// gate-test-abi_vectorcall
//@ add-core-stubs
//@ needs-llvm-components: x86
//@ compile-flags: --target=i686-pc-windows-msvc --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;
use minicore::*;

// Test that the "vectorcall" ABI is feature-gated, and cannot be used when
// the `vectorcall` feature gate is not used.

extern "vectorcall" fn f() {} //~ ERROR "vectorcall" ABI is experimental

trait T {
    extern "vectorcall" fn m(); //~ ERROR "vectorcall" ABI is experimental

    extern "vectorcall" fn dm() {} //~ ERROR "vectorcall" ABI is experimental
}

struct S;
impl T for S {
    extern "vectorcall" fn m() {} //~ ERROR "vectorcall" ABI is experimental
}

impl S {
    extern "vectorcall" fn im() {} //~ ERROR "vectorcall" ABI is experimental
}

type TA = extern "vectorcall" fn(); //~ ERROR "vectorcall" ABI is experimental

extern "vectorcall" {} //~ ERROR "vectorcall" ABI is experimental
