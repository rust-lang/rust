//@ check-pass
#![deny(improper_ctypes)]

//@ aux-build: extern_crate_types.rs
//@ compile-flags:--extern extern_crate_types
extern crate extern_crate_types as ext_crate;


// Issue: https://github.com/rust-lang/rust/issues/132699
// FFI-safe pointers to nonexhaustive structs should be FFI-safe too

// BEGIN: this is the exact same code as in ext_crate, to compare the lints
#[repr(C)]
#[non_exhaustive]
pub struct OtherNonExhaustiveStruct {
    pub field: u8
}

extern "C" {
    pub fn othernonexhaustivestruct_create() -> *mut OtherNonExhaustiveStruct;
    pub fn othernonexhaustivestruct_destroy(s: *mut OtherNonExhaustiveStruct);
}
// END

use ext_crate::NonExhaustiveStruct;

extern "C" {
    pub fn use_struct(s: *mut NonExhaustiveStruct);
}

fn main() {}
