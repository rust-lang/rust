//@ add-minicore
//@ compile-flags: --target aarch64-unknown-linux-gnu -Zinline-mir=no -C no-prepopulate-passes -Copt-level=3
//@ needs-llvm-components: aarch64
//@ min-llvm-version: 23

#![crate_type = "lib"]
#![feature(no_core, lang_items)]
#![no_core]

extern crate minicore;
use minicore::*;

#[inline(always)]
#[target_feature(enable = "neon")]
#[no_mangle]
pub fn single_target_feature() -> i32 {
    42
}

#[inline(always)]
#[target_feature(enable = "neon,i8mm")]
#[no_mangle]
// CHECK: define{{( noundef)?}} i32 @multiple_target_features() unnamed_addr #1{{( !guid ![0-9]+)?}} {
pub fn multiple_target_features() -> i32 {
    // CHECK: %_0 = call{{( noundef)?}} i32 @single_target_feature() #3
    single_target_feature()
}

#[no_mangle]
// CHECK: define{{( noundef)?}} i32 @inherits_from_global() unnamed_addr #2{{( !guid ![0-9]+)?}} {
pub fn inherits_from_global() -> i32 {
    unsafe {
        // CHECK: %_0 = call{{( noundef)?}} i32 @single_target_feature() #3
        single_target_feature()
    }
}
