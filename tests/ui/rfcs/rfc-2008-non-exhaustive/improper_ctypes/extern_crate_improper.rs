// aux-build:types.rs
#![deny(improper_ctypes)]

extern crate types;

// This test checks that non-exhaustive types with `#[repr(C)]` from an extern crate are considered
// improper.

use types::{NonExhaustiveEnum, NonExhaustiveVariants, NormalStruct, TupleStruct, UnitStruct};

extern "C" {
    pub fn non_exhaustive_enum(_: NonExhaustiveEnum);
    //~^ ERROR `extern` block uses type `NonExhaustiveEnum`, which is not FFI-safe
    pub fn non_exhaustive_normal_struct(_: NormalStruct);
    //~^ ERROR `extern` block uses type `NormalStruct`, which is not FFI-safe
    pub fn non_exhaustive_unit_struct(_: UnitStruct);
    //~^ ERROR `extern` block uses type `UnitStruct`, which is not FFI-safe
    pub fn non_exhaustive_tuple_struct(_: TupleStruct);
    //~^ ERROR `extern` block uses type `TupleStruct`, which is not FFI-safe
    pub fn non_exhaustive_variant(_: NonExhaustiveVariants);
    //~^ ERROR `extern` block uses type `NonExhaustiveVariants`, which is not FFI-safe
}

fn main() {}
