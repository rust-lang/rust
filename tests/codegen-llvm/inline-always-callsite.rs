//@ add-minicore
//@ compile-flags: --target aarch64-unknown-linux-gnu -Zinline-mir=no -C no-prepopulate-passes
//@ needs-llvm-components: aarch64

#![crate_type = "lib"]
#![feature(no_core, lang_items, target_feature_inline_always)]
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
// CHECK: define{{( noundef)?}} i32 @multiple_target_features() unnamed_addr #1 {
pub fn multiple_target_features() -> i32 {
    // CHECK: %_0 = call{{( noundef)?}} i32 @single_target_feature() #3
    single_target_feature()
}

#[no_mangle]
// CHECK: define{{( noundef)?}} i32 @inherits_from_global() unnamed_addr #2 {
pub fn inherits_from_global() -> i32 {
    unsafe {
        // CHECK: %_0 = call{{( noundef)?}} i32 @single_target_feature() #3
        single_target_feature()
    }
}

// Attribute #3 requires the alwaysinline attribute, the alwaysinline attribute is not emitted on a
// function definition when target features are present, rather it will be moved onto the function
// call, if the features match up.
//
// CHECK: attributes #3 = { alwaysinline nounwind }
