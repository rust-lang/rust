//@ add-core-stubs
//@ needs-llvm-components: nvptx
//@ compile-flags: --target=nvptx64-nvidia-cuda --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]

extern crate minicore;
use minicore::*;

extern "ptx-kernel" fn fu() {} //~ ERROR extern "ptx-kernel" ABI is experimental

trait T {
    extern "ptx-kernel" fn mu(); //~ ERROR extern "ptx-kernel" ABI is experimental
    extern "ptx-kernel" fn dmu() {} //~ ERROR extern "ptx-kernel" ABI is experimental
}

struct S;
impl T for S {
    extern "ptx-kernel" fn mu() {} //~ ERROR extern "ptx-kernel" ABI is experimental
}

impl S {
    extern "ptx-kernel" fn imu() {} //~ ERROR extern "ptx-kernel" ABI is experimental
}

type TAU = extern "ptx-kernel" fn(); //~ ERROR extern "ptx-kernel" ABI is experimental

extern "ptx-kernel" {} //~ ERROR extern "ptx-kernel" ABI is experimental
