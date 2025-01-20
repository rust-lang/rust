#![feature(rustc_attrs)]
#![allow(dead_code)]

// See also: repr-c-dead-variants.rs

//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$SOME_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"

// A simple uninhabited type.
enum Void {}

// Compiler must not remove dead variants of `#[repr(C, int)]` ADTs.
#[repr(C, u8)]
#[rustc_layout(debug)]
enum UnivariantU8 { //~ ERROR layout_of
    Variant(Void),
}

// ADTs with variants that have fields must have space allocated for those fields.
#[repr(C, u8)]
#[rustc_layout(debug)]
enum TwoVariantsU8 { //~ ERROR layout_of
    Variant1(Void),
    Variant2(u8),
}

// Some targets have 4-byte-aligned u64, make it always 8-byte-aligned.
#[repr(C, align(8))]
struct Align8U64(u64);

// This one is 2 x u64: we reserve space for fields in a dead branch.
#[repr(C, u8)]
#[rustc_layout(debug)]
enum DeadBranchHasOtherFieldU8 { //~ ERROR layout_of
    Variant1(Void, Align8U64),
    Variant2(u8),
}

fn main() {}
