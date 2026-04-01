//@ only-x86_64
//@ normalize-stderr: "(abi|pref|unadjusted_abi_align): Align\([1-8] bytes\)" -> "$1: $$SOME_ALIGN"

#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[rustc_abi(debug)]
extern "sysv64" fn pass_zst(_: ()) {} //~ ERROR: fn_abi
