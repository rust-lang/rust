// gate-test-c_unwind
// needs-llvm-components: x86
// compile-flags: --target=i686-pc-windows-msvc --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]
#[lang="sized"]
trait Sized { }

// Test that the "stdcall-unwind" ABI is feature-gated, and cannot be used when
// the `c_unwind` feature gate is not used.

extern "stdcall-unwind" fn fu() {} //~ ERROR stdcall-unwind ABI is experimental

trait T {
    extern "stdcall-unwind" fn mu(); //~ ERROR stdcall-unwind ABI is experimental
    extern "stdcall-unwind" fn dmu() {} //~ ERROR stdcall-unwind ABI is experimental
}

struct S;
impl T for S {
    extern "stdcall-unwind" fn mu() {} //~ ERROR stdcall-unwind ABI is experimental
}

impl S {
    extern "stdcall-unwind" fn imu() {} //~ ERROR stdcall-unwind ABI is experimental
}

type TAU = extern "stdcall-unwind" fn(); //~ ERROR stdcall-unwind ABI is experimental

extern "stdcall-unwind" {} //~ ERROR stdcall-unwind ABI is experimental
