//@ add-core-stubs
//@ compile-flags: --target thumbv8m.main-none-eabihf
//@ normalize-stderr: "randomization_seed: \d+" -> "randomization_seed: $$SEED"
//@ needs-llvm-components: arm
//
// Verify that thumb targets implement the repr(C) for enums correctly.
//
// See #87917
#![feature(never_type, rustc_attrs, no_core, lang_items)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

#[rustc_layout(debug)]
#[repr(C)]
enum A { Apple } //~ ERROR: layout_of

#[rustc_layout(debug)]
#[repr(C)]
enum B { Banana = 255, } //~ ERROR: layout_of

#[rustc_layout(debug)]
#[repr(C)]
enum C { Chaenomeles = 256, } //~ ERROR: layout_of

#[rustc_layout(debug)]
#[repr(C)]
enum P { Peach = 0x1000_0000isize, } //~ ERROR: layout_of

const TANGERINE: usize = 0x8100_0000; // hack to get negative numbers without negation operator!

#[rustc_layout(debug)]
#[repr(C)]
enum T { Tangerine = TANGERINE as isize } //~ ERROR: layout_of
