// Rust side of the IR-level interop test, mirroring the C functions in `interop.c`.

#![feature(core_intrinsics)]
#![allow(internal_features)]

use std::ffi::{c_char, c_int, c_void};
use std::intrinsics::ptrauth_c_cast;
use std::mem::transmute;

type BinOp = unsafe extern "C" fn(c_int, c_int) -> c_int;
type GenericCb = unsafe extern "C" fn();

unsafe extern "C" {
    fn apply(op: BinOp, a: c_int, b: c_int) -> c_int;
    fn get_sub() -> BinOp;
    fn lookup(name: *const c_char) -> *mut c_void;
    fn register_callback(cb: Option<GenericCb>);
    fn get_proc(name: *const c_char) -> Option<GenericCb>;
    fn call_opaque(f: *mut c_void, a: c_int, b: c_int) -> c_int;
}

pub extern "C" fn add(a: c_int, b: c_int) -> c_int {
    a + b
}

// Correctly typed callback, Rust to C: `add` is signed with the discriminator C authenticates with
// in `apply`.
#[no_mangle]
pub unsafe fn typed_to_c() -> c_int {
    unsafe { apply(add, 2, 3) }
}

// Correctly typed function pointer, C to Rust: called with the discriminator C signed it with in
// `get_sub`.
#[no_mangle]
pub unsafe fn typed_from_c() -> c_int {
    unsafe { get_sub()(9, 4) }
}

// `void *` from C (signed with 0 in `lookup`).
#[no_mangle]
pub unsafe fn from_void_ptr() -> Option<BinOp> {
    unsafe { ptrauth_c_cast(lookup(c"mul".as_ptr())) }
}

// Generic callback slot, Rust to C (authenticated as `generic_cb_t` by the cast in
// `invoke_callback`).
#[no_mangle]
pub unsafe fn to_generic_slot() {
    unsafe { register_callback(Some(ptrauth_c_cast::<BinOp, GenericCb>(add))) }
}

// Generic callback from C (signed as `generic_cb_t` in `get_proc`).
#[no_mangle]
pub unsafe fn from_generic_slot() -> Option<BinOp> {
    unsafe { ptrauth_c_cast(get_proc(c"sub".as_ptr())) }
}

// `void *` user data, Rust to C (authenticated with 0 by the cast in `call_opaque`).
#[no_mangle]
pub unsafe fn to_void_ptr() -> c_int {
    unsafe { call_opaque(ptrauth_c_cast::<BinOp, *mut c_void>(add), 4, 5) }
}

// A Rust-only round trip through another type needs no resign.
#[no_mangle]
pub unsafe fn rust_round_trip() -> c_int {
    unsafe {
        let erased: GenericCb = transmute::<BinOp, GenericCb>(add);
        let back: BinOp = transmute(erased);
        back(1, 1)
    }
}
