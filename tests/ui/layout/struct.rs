//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$PREF_ALIGN"
//! Various struct layout tests.

#![feature(rustc_attrs)]
#![feature(never_type)]
#![crate_type = "lib"]

#[rustc_dump_layout(backend_repr)]
struct AlignedZstPreventsScalar(i16, [i32; 0]); //~ERROR: backend_repr: Memory

#[rustc_dump_layout(backend_repr)]
struct AlignedZstButStillScalar(i32, [i16; 0]); //~ERROR: backend_repr: Scalar
