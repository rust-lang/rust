// Checks if the correct annotation for the sysv64 ABI is passed to
// llvm. Also checks that the abi-sysv64 feature gate allows usage
// of the sysv64 abi.
//
//@ add-core-stubs
//@ needs-llvm-components: x86
//@ compile-flags: -C no-prepopulate-passes --target=x86_64-unknown-linux-gnu -Copt-level=0

#![crate_type = "lib"]
#![no_core]
#![feature(abi_x86_interrupt, no_core, lang_items)]

extern crate minicore;
use minicore::*;

// CHECK: define x86_64_sysvcc i64 @has_sysv64_abi
#[no_mangle]
pub extern "sysv64" fn has_sysv64_abi(a: i64) -> i64 {
    a
}
