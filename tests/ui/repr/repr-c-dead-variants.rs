#![feature(no_core, rustc_attrs, lang_items)]
#![allow(dead_code)]
#![crate_type = "lib"]
#![no_std]
#![no_core]

extern crate minicore;
use minicore::*;

// See also: repr-c-int-dead-variants.rs

//@ add-core-stubs
//@ normalize-stderr: "pref: Align\([1-8] bytes\)" -> "pref: $$SOME_ALIGN"
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"

// This test depends on the value of the `c_enum_min_bits` target option.
// As there's no way to actually check it from UI test, we only run this test on a subset of archs.
// Four archs specifically are chosen: one for major architectures (x86_64, i686, aarch64)
// and `armebv7r-none-eabi` that has `c_enum_min_bits` set to 8.

//@ revisions: aarch64-unknown-linux-gnu
//@[aarch64-unknown-linux-gnu] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64-unknown-linux-gnu] needs-llvm-components: aarch64

//@ revisions: i686-pc-windows-msvc
//@[i686-pc-windows-msvc] compile-flags: --target i686-pc-windows-gnu
//@[i686-pc-windows-msvc] needs-llvm-components: x86

//@ revisions: x86_64-unknown-linux-gnu
//@[x86_64-unknown-linux-gnu] compile-flags: --target x86_64-unknown-linux-gnu
//@[x86_64-unknown-linux-gnu] needs-llvm-components: x86
//
//@ revisions: armebv7r-none-eabi
//@[armebv7r-none-eabi] compile-flags: --target armebv7r-none-eabi
//@[armebv7r-none-eabi] needs-llvm-components: arm

// A simple uninhabited type.
enum Void {}

// Compiler must not remove dead variants of `#[repr(C, int)]` ADTs.
#[repr(C)]
#[rustc_layout(debug)]
enum Univariant { //~ ERROR layout_of
    Variant(Void),
}

// ADTs with variants that have fields must have space allocated for those fields.
#[repr(C)]
#[rustc_layout(debug)]
enum TwoVariants { //~ ERROR layout_of
    Variant1(Void),
    Variant2(u8),
}

// Some targets have 4-byte-aligned u64, make it always 8-byte-aligned.
#[repr(C, align(8))]
struct Align8U64(u64);

// This one is 2 x u64: we reserve space for fields in a dead branch.
#[repr(C)]
#[rustc_layout(debug)]
enum DeadBranchHasOtherField { //~ ERROR layout_of
    Variant1(Void, Align8U64),
    Variant2(u8),
}
