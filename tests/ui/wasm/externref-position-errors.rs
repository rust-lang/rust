//! wasm `externref` is a bare-position-only type: legal as a function
//! parameter, return value or local (including function pointer signature
//! slots), and nowhere else. All violations are type-check-time errors.

//@ add-minicore
//@ compile-flags: --target wasm32-unknown-unknown
//@ needs-llvm-components: webassembly
//@ ignore-backends: gcc
//@ check-fail

#![crate_type = "lib"]
#![no_std]
#![no_core]
#![feature(no_core, lang_items)]
#![allow(non_camel_case_types)]

extern crate minicore;
use minicore::*;

#[lang = "externref"]
#[non_exhaustive]
pub struct externref;

impl Copy for externref {}

extern "C" {
    fn create_ref() -> externref; // OK: bare return slot
    fn use_ref(v: externref); // OK: bare parameter slot
    fn bad_ref_param(v: &externref);
    //~^ ERROR wasm `externref` cannot be used inside `&externref`
    static BAD_STATIC: externref;
    //~^ ERROR wasm `externref` cannot be used in a `static`
}

pub struct BadField {
    pub v: externref,
    //~^ ERROR wasm `externref` cannot be used in a struct field
}

// Even single-field wrappers are rejected: fields are not value slots.
pub struct BadWrapper(externref);
//~^ ERROR wasm `externref` cannot be used in a struct field

pub extern "C" fn ok_identity(v: externref) -> externref {
    v
}

pub fn ok_locals() {
    let v = unsafe { create_ref() };
    let w = v; // Copy
    unsafe { use_ref(w) };
    unsafe { use_ref(v) };
}

pub fn ok_fn_ptr(v: externref) -> externref {
    let f: extern "C" fn(externref) -> externref = ok_identity;
    f(v)
}

pub fn bad_tuple(v: externref) {
    let t = (v, v);
    //~^ ERROR wasm `externref` cannot be used inside `(externref, externref)`
    //~| ERROR wasm `externref` cannot be used inside `(externref, externref)`
}

pub fn bad_array(v: externref) {
    let a = [v, v];
    //~^ ERROR wasm `externref` cannot be used inside `[externref; 2]`
    //~| ERROR wasm `externref` cannot be used inside `[externref; 2]`
}

pub fn bad_borrow(v: externref) {
    let r = &v;
    //~^ ERROR wasm `externref` cannot be used inside `&externref`
    //~| ERROR wasm `externref` cannot be used inside `&externref`
}

pub fn generic<T>(t: T) -> T {
    t
}

pub fn bad_generic_arg(v: externref) {
    generic(v);
    //~^ ERROR wasm `externref` cannot be used as a generic argument
}

pub fn bad_closure_capture(v: externref) {
    let _c = move || v;
    //~^ ERROR wasm `externref` cannot be used inside
    //~| ERROR wasm `externref` cannot be used inside
}
