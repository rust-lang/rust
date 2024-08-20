//@ revisions: other other-linux x86_64-pc-windows-gnu s390x-linux sparc64-linux powerpc-linux
//@ normalize-stderr-test: "(abi|pref|unadjusted_abi_align): Align\([1-8] bytes\)" -> "$1: $$SOME_ALIGN"
// ZSTs are only not ignored when the target_env is "gnu", "musl" or "uclibc". However, Rust does
// not currently support any other target_env on these architectures.

// Ignore the ZST revisions
//@[other] ignore-x86_64-pc-windows-gnu
//@[other] ignore-linux
//@[other-linux] only-linux
//@[other-linux] ignore-s390x
//@[other-linux] ignore-sparc64
//@[other-linux] ignore-powerpc

// Pass the ZST indirectly revisions
//@[x86_64-pc-windows-gnu] only-x86_64-pc-windows-gnu
//@[s390x-linux] only-s390x
//@[s390x-linux] only-linux
//@[sparc64-linux] only-sparc64
//@[sparc64-linux] only-linux
//@[powerpc-linux] only-powerpc
//@[powerpc-linux] only-linux

#![feature(rustc_attrs)]
#![crate_type = "lib"]

#[rustc_abi(debug)]
extern "C" fn pass_zst(_: ()) {} //~ ERROR: fn_abi
