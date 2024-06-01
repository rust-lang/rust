//@ only-x86_64
//@ revisions: other windows-gnu
//@[other] ignore-windows-gnu
//@[windows-gnu] only-windows-gnu

#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[rustc_abi(debug)]
extern "win64" fn pass_zst(_: ()) {} //~ ERROR: fn_abi
