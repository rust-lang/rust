//@ add-minicore
//@ needs-llvm-components: x86
//@ compile-flags: --target=x86_64-unknown-linux-gnu --crate-type=rlib
//@ ignore-backends: gcc
#![no_core]
#![feature(no_core, lang_items, abi_swift)]

extern crate minicore;
use minicore::*;

// The Swift ABI is only stable on Apple platforms, so it must be rejected
// on other targets even when the `abi_swift` feature gate is enabled.

extern "Swift" fn f() {} //~ ERROR is not a supported ABI

trait T {
    extern "Swift" fn m(); //~ ERROR is not a supported ABI

    extern "Swift" fn dm() {} //~ ERROR is not a supported ABI
}

struct S;
impl T for S {
    extern "Swift" fn m() {} //~ ERROR is not a supported ABI
}

impl S {
    extern "Swift" fn im() {} //~ ERROR is not a supported ABI
}

type TA = extern "Swift" fn(); //~ ERROR is not a supported ABI

extern "Swift" {} //~ ERROR is not a supported ABI
