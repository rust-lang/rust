//@ add-minicore
//@ min-llvm-version: 24
//@ revisions: armhf armsf aarch64sf

//@ [armhf] needs-llvm-components: arm
//@ [armhf] compile-flags: --target=armv7-unknown-linux-gnueabihf

//@ [armsf] needs-llvm-components: arm
//@ [armsf] compile-flags: --target=armv7-unknown-linux-gnueabi

//@ [aarch64sf] needs-llvm-components: aarch64
//@ [aarch64sf] compile-flags: --target=aarch64-unknown-none-softfloat

#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

// rustc sets the module flag only for targets with an explicit `llvm_floatabi`,
// which is (currently) only the case on ARM.

// armhf: !{i32 1, !"float-abi", !"hard"}
// armsf: !{i32 1, !"float-abi", !"soft"}
// aarch64sf-NOT: !"float-abi"
