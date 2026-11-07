//@ add-minicore
//@ compile-flags: -Zmir-opt-level=0

// Verify that a `#![no_core]` crate that does not define the `raw_eq` and
// `compare_bytes` lang items still compares an array pattern expanded from a
// constant element by element, rather than failing to find the intrinsics.

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

// Constant patterns require `PartialEq` and `StructuralPartialEq`, which
// minicore does not provide. Only the MIR is checked, so the impls do not need
// meaningful bodies.
#[lang = "eq"]
pub trait PartialEq<Rhs = Self> {
    fn eq(&self, other: &Rhs) -> bool;
}

#[lang = "structural_peq"]
pub trait StructuralPartialEq {}

impl PartialEq for u8 {
    fn eq(&self, _: &u8) -> bool {
        true
    }
}

impl PartialEq for [u8; 4] {
    fn eq(&self, _: &[u8; 4]) -> bool {
        true
    }
}

// EMIT_MIR aggregate_array_eq_no_core.array_match.built.after.mir
pub fn array_match(x: [u8; 4]) -> bool {
    // CHECK-LABEL: fn array_match(
    // CHECK-NOT: raw_eq
    // CHECK: switchInt(copy _1[0 of 4])
    // CHECK: switchInt(copy _1[1 of 4])
    // CHECK: switchInt(copy _1[2 of 4])
    // CHECK: switchInt(copy _1[3 of 4])
    const EXPECTED: [u8; 4] = [1, 2, 3, 4];
    match x {
        EXPECTED => true,
        _ => false,
    }
}
