//@ only-wasm32-wasip1
//@ compile-flags: --crate-type=lib
//@ build-pass
//@ aux-build:internal_unstable.rs

#[macro_use]
extern crate internal_unstable;

asm_redirect!(
    "test:",
    ".globl test",
    ".functype test (i32) -> (i32)",
    "local.get 0",
    "i32.const 1",
    "i32.add",
    "end_function",
    ".export_name test, test",
);
