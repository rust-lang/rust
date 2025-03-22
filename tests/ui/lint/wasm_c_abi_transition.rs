//@ compile-flags: --target wasm32-unknown-unknown
//@ needs-llvm-components: webassembly
//@ add-core-stubs
//@ build-fail

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]
#![deny(wasm_c_abi)]

extern crate minicore;
use minicore::*;

pub extern "C" fn my_fun_trivial(_x: i32, _y: f32) {}

#[repr(C)]
pub struct MyType(i32, i32);
pub extern "C" fn my_fun(_x: MyType) {} //~ERROR: wasm ABI transition
//~^WARN: previously accepted

// This one is ABI-safe as it only wraps a single field,
// and the return type can be anything.
#[repr(C)]
pub struct MySafeType(i32);
pub extern "C" fn my_fun_safe(_x: MySafeType) -> MyType { loop {} }

// This one not ABI-safe due to the alignment.
#[repr(C, align(16))]
pub struct MyAlignedType(i32);
pub extern "C" fn my_fun_aligned(_x: MyAlignedType) {} //~ERROR: wasm ABI transition
//~^WARN: previously accepted

// Check call-site warning
extern "C" {
    fn other_fun(x: MyType);
}

pub fn call_other_fun(x: MyType) {
    unsafe { other_fun(x) } //~ERROR: wasm ABI transition
    //~^WARN: previously accepted
}
