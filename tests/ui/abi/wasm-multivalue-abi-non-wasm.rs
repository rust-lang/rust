//@ ignore-wasm32
//@ compile-flags: --crate-type rlib

#![feature(abi_wasm_multivalue)]

extern "wasm-multivalue" fn f() {} //~ ERROR is not a supported ABI

trait T {
    extern "wasm-multivalue" fn m(); //~ ERROR is not a supported ABI

    extern "wasm-multivalue" fn dm() {} //~ ERROR is not a supported ABI
}

struct S;
impl T for S {
    extern "wasm-multivalue" fn m() {} //~ ERROR is not a supported ABI
}

impl S {
    extern "wasm-multivalue" fn im() {} //~ ERROR is not a supported ABI
}

type TA = extern "wasm-multivalue" fn(); //~ ERROR is not a supported ABI

extern "wasm-multivalue" {} //~ ERROR is not a supported ABI
