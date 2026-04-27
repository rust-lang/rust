//@ add-minicore
//@ needs-llvm-components: x86
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
//@ ignore-backends: gcc
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;
use minicore::*;

// Test that the "Swift" ABI is feature-gated, and cannot be used when
// the `abi_swift` feature gate is not used.

extern "Swift" fn f() {} //~ ERROR "Swift" ABI is experimental

trait T {
    extern "Swift" fn m(); //~ ERROR "Swift" ABI is experimental

    extern "Swift" fn dm() {} //~ ERROR "Swift" ABI is experimental
}

struct S;
impl T for S {
    extern "Swift" fn m() {} //~ ERROR "Swift" ABI is experimental
}

impl S {
    extern "Swift" fn im() {} //~ ERROR "Swift" ABI is experimental
}

type TA = extern "Swift" fn(); //~ ERROR "Swift" ABI is experimental

extern "Swift" {} //~ ERROR "Swift" ABI is experimental
