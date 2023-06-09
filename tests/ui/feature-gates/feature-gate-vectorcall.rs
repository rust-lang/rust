// gate-test-abi_vectorcall
// needs-llvm-components: x86
// compile-flags: --target=i686-pc-windows-msvc --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]
#[lang="sized"]
trait Sized { }

// Test that the "vectorcall" ABI is feature-gated, and cannot be used when
// the `vectorcall` feature gate is not used.

extern "vectorcall" fn f() {} //~ ERROR vectorcall is experimental

trait T {
    extern "vectorcall" fn m(); //~ ERROR vectorcall is experimental

    extern "vectorcall" fn dm() {} //~ ERROR vectorcall is experimental
}

struct S;
impl T for S {
    extern "vectorcall" fn m() {} //~ ERROR vectorcall is experimental
}

impl S {
    extern "vectorcall" fn im() {} //~ ERROR vectorcall is experimental
}

type TA = extern "vectorcall" fn(); //~ ERROR vectorcall is experimental

extern "vectorcall" {} //~ ERROR vectorcall is experimental
