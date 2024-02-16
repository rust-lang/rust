//@ needs-llvm-components: webassembly
//@ compile-flags: --target=wasm32-unknown-unknown --crate-type=rlib
#![no_core]
#![feature(no_core, lang_items)]
#[lang="sized"]
trait Sized { }

extern "wasm" fn fu() {} //~ ERROR wasm ABI is experimental

trait T {
    extern "wasm" fn mu(); //~ ERROR wasm ABI is experimental
    extern "wasm" fn dmu() {} //~ ERROR wasm ABI is experimental
}

struct S;
impl T for S {
    extern "wasm" fn mu() {} //~ ERROR wasm ABI is experimental
}

impl S {
    extern "wasm" fn imu() {} //~ ERROR wasm ABI is experimental
}

type TAU = extern "wasm" fn(); //~ ERROR wasm ABI is experimental

extern "wasm" {} //~ ERROR wasm ABI is experimental
