//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$PREF_ALIGN"
//! Various struct layout tests.

#![feature(rustc_attrs)]
#![feature(never_type)]
#![crate_type = "lib"]

#[rustc_layout(abi)]
struct AlignedZstPreventsScalar(i16, [i32; 0]); //~ERROR: abi: Memory

#[rustc_layout(abi)]
struct AlignedZstButStillScalar(i32, [i16; 0]); //~ERROR: abi: Scalar
