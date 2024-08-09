//@ only-x86_64
//@ revisions: other windows-gnu
//@ normalize-stderr-test: "(abi|pref|unadjusted_abi_align): Align\([1-8] bytes\)" -> "$1: $$SOME_ALIGN"
//@[other] ignore-windows-gnu
//@[windows-gnu] only-windows-gnu

#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[rustc_abi(debug)]
extern "win64" fn pass_zst(_: ()) {} //~ ERROR: fn_abi
