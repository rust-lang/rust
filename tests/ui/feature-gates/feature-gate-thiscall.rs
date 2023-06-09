// gate-test-abi_thiscall
// needs-llvm-components: x86
// compile-flags: --target=i686-pc-windows-msvc --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]
#[lang="sized"]
trait Sized { }

// Test that the "thiscall" ABI is feature-gated, and cannot be used when
// the `abi_thiscall` feature gate is not used.

extern "thiscall-unwind" fn fu() {} //~ ERROR thiscall-unwind ABI is experimental
extern "thiscall" fn f() {} //~ ERROR thiscall is experimental

trait T {
    extern "thiscall" fn m(); //~ ERROR thiscall is experimental
    extern "thiscall-unwind" fn mu(); //~ ERROR thiscall-unwind ABI is experimental

    extern "thiscall" fn dm() {} //~ ERROR thiscall is experimental
    extern "thiscall-unwind" fn dmu() {} //~ ERROR thiscall-unwind ABI is experimental
}

struct S;
impl T for S {
    extern "thiscall" fn m() {} //~ ERROR thiscall is experimental
    extern "thiscall-unwind" fn mu() {} //~ ERROR thiscall-unwind ABI is experimental
}

impl S {
    extern "thiscall" fn im() {} //~ ERROR thiscall is experimental
    extern "thiscall-unwind" fn imu() {} //~ ERROR thiscall-unwind ABI is experimental
}

type TA = extern "thiscall" fn(); //~ ERROR thiscall is experimental
type TAU = extern "thiscall-unwind" fn(); //~ ERROR thiscall-unwind ABI is experimental

extern "thiscall" {} //~ ERROR thiscall is experimental
extern "thiscall-unwind" {} //~ ERROR thiscall-unwind ABI is experimental
