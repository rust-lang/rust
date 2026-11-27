//@ add-minicore
//@ needs-llvm-components: aarch64
//@ revisions: ARM64E ARM64 PAUTHTEST
//@ compile-flags: -Copt-level=0 -Cpanic=abort
//@ [ARM64E] compile-flags: --target=arm64e-apple-ios
//@ [ARM64] compile-flags: --target=aarch64-apple-ios
//@ [PAUTHTEST] compile-flags: --target=aarch64-unknown-linux-pauthtest

#![feature(no_core)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;

// CHECK-LABEL: define {{.*}} @abi_version(
#[no_mangle]
pub extern "C" fn abi_version() {}

// ARM64E: !{{[0-9]+}} = !{i32 6, !"ptrauth.abi-version", ![[VERSIONS:[0-9]+]]}
// ARM64E: ![[VERSIONS]] = !{![[VERSION:[0-9]+]]}
// ARM64E: ![[VERSION]] = !{i32 0, i1 false}
// ARM64-NOT: ptrauth.abi-version
// PAUTHTEST-NOT: ptrauth.abi-version
