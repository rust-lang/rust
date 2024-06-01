//@ only-x86_64

#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[rustc_abi(debug)]
extern "sysv64" fn pass_zst(_: ()) {} //~ ERROR: fn_abi
