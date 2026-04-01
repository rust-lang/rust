//@ add-minicore
//@ build-fail
#![no_core]
#![crate_type = "lib"]
#![feature(no_core, unsized_fn_params)]
#![allow(improper_ctypes_definitions, improper_ctypes)]

extern crate minicore;
use minicore::*;

fn rust(_: [u8]) {}
extern "C" fn c(_: [u8]) {}
//~^ ERROR this function definition uses unsized type `[u8]` which is not supported with the chosen ABI
extern "system" fn system(_: [u8]) {}
//~^ ERROR this function definition uses unsized type `[u8]` which is not supported with the chosen ABI

#[repr(C)]
struct CustomUnsized {
    a: i64,
    b: [u8],
}

extern "C" fn c_custom_unsized(x: CustomUnsized) {}
//~^ ERROR this function definition uses unsized type `CustomUnsized` which is not supported with the chosen ABI

#[unsafe(no_mangle)]
fn entry(x: [u8], y: [u8], z: [u8], w: CustomUnsized) {
    rust(x);
    c(y);
    //~^ ERROR this function call uses unsized type `[u8]` which is not supported with the chosen ABI
    system(z);
    //~^ ERROR this function call uses unsized type `[u8]` which is not supported with the chosen ABI
    c_custom_unsized(w);
    //~^ ERROR this function call uses unsized type `CustomUnsized` which is not supported with the chosen ABI
}

#[unsafe(no_mangle)]
fn test_fn_ptr(rust: extern "Rust" fn(_: [u8]), c: extern "C" fn(_: [u8]), x: [u8], y: [u8]) {
    rust(x);
    c(y);
    //~^ ERROR this function call uses unsized type `[u8]` which is not supported with the chosen ABI
}

#[unsafe(no_mangle)]
fn test_extern(x: [u8], y: [u8]) {
    unsafe extern "Rust" {
        safe fn rust(_: [u8]);
    }

    unsafe extern "system" {
        safe fn system(_: [u8]);
    }

    rust(x);
    system(y);
    //~^ ERROR this function call uses unsized type `[u8]` which is not supported with the chosen ABI
}

extern "C" fn c_polymorphic<T: ?Sized>(_: T) {}
//~^ ERROR this function definition uses unsized type `[u8]` which is not supported with the chosen ABI

#[unsafe(no_mangle)]
fn test_polymorphic(x: [u8]) {
    c_polymorphic(x);
    //~^ ERROR this function call uses unsized type `[u8]` which is not supported with the chosen ABI
}
