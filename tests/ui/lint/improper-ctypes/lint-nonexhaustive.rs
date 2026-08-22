#![deny(improper_ctypes)]

//@ aux-build: extern_crate_types.rs
//@ compile-flags:--extern extern_crate_types
extern crate extern_crate_types as ext_crate;

// Properly deal with non_exhaustive types:
// the thorny logic is expressed in https://github.com/rust-lang/rust/issues/44109#issuecomment-537583344
// and its linked comments
//

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

//FIXME these tests feel almost pointless given they concern types that need
// to be behind indirections... but soon.
pub extern "C" fn fnptr_wrapper(
    _use_struct: extern "C" fn(s: ext_crate::NonExhaustiveStruct),
    //~^ ERROR `extern` callback uses type `NonExhaustiveStruct`

    _use_enum: extern "C" fn(s: ext_crate::NonExhaustiveEnum),
    //~^ ERROR `extern` callback uses type `NonExhaustiveEnum`
    _use_enumwbv: extern "C" fn(s: ext_crate::NonExhaustiveEnumWithBadVariant),
    //~^ ERROR `extern` callback uses type `NonExhaustiveEnumWithBadVariant`
    _use_pureenum: extern "C" fn(s: ext_crate::NonExhaustivePureEnum),
    _use_optionlike: extern "C" fn(s: ext_crate::NonExhaustiveOptionLike),
    //~^ ERROR `extern` callback uses type `NonExhaustiveOptionLike`
){}

fn main() {}
