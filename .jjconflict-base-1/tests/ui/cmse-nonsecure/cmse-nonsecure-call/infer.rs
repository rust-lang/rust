//@ add-minicore
//@ compile-flags: --target thumbv8m.main-none-eabi --crate-type lib
//@ needs-llvm-components: arm
//@ ignore-backends: gcc
#![feature(abi_cmse_nonsecure_call, no_core, lang_items)]
#![no_core]

// Infer variables cause panics in layout generation, so the argument/return type is checked for
// whether it contains an infer var, and `LayoutError::Unknown` is emitted if so.
//
// See https://github.com/rust-lang/rust/issues/130104.

extern crate minicore;
use minicore::*;

fn infer_1() {
    let _ = mem::transmute::<fn() -> _, extern "cmse-nonsecure-call" fn() -> _>;
    //~^ ERROR type annotations needed
}

fn infer_2() {
    let _ = mem::transmute::<fn() -> (i32, _), extern "cmse-nonsecure-call" fn() -> (i32, _)>;
    //~^ ERROR type annotations needed
}

fn infer_3() {
    let _ = mem::transmute::<fn(_: _) -> (), extern "cmse-nonsecure-call" fn(_: _) -> ()>;
    //~^ ERROR type annotations needed
}

fn infer_4() {
    let _ =
        mem::transmute::<fn(_: (i32, _)) -> (), extern "cmse-nonsecure-call" fn(_: (i32, _)) -> ()>;
    //~^ ERROR type annotations needed
}
