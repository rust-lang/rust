//@ only-wasm32
//@ compile-flags: --crate-type rlib

// Test that the "wasm-multivalue" ABI is feature-gated, and cannot be used
// when the `abi_wasm_multivalue` feature gate is not used.

extern "wasm-multivalue" fn f() {} //~ ERROR "wasm-multivalue" ABI is experimental

trait T {
    extern "wasm-multivalue" fn m(); //~ ERROR "wasm-multivalue" ABI is experimental

    extern "wasm-multivalue" fn dm() {} //~ ERROR "wasm-multivalue" ABI is experimental
}

struct S;
impl T for S {
    extern "wasm-multivalue" fn m() {} //~ ERROR "wasm-multivalue" ABI is experimental
}

impl S {
    extern "wasm-multivalue" fn im() {} //~ ERROR "wasm-multivalue" ABI is experimental
}

type TA = extern "wasm-multivalue" fn(); //~ ERROR "wasm-multivalue" ABI is experimental

extern "wasm-multivalue" {} //~ ERROR "wasm-multivalue" ABI is experimental
