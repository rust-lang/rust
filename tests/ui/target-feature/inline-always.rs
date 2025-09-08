//@ add-core-stubs
//@ build-pass
//@ compile-flags: --crate-type=lib
//@ revisions: aarch64
//@[aarch64] compile-flags: --target aarch64-unknown-linux-gnu
//@[aarch64] needs-llvm-components: aarch64

#![feature(no_core, target_feature_inline_always)]
#![no_core]

extern crate minicore;
use minicore::*;

#[inline(always)]
#[target_feature(enable = "neon,fp16")]
pub unsafe fn target_feature_identity() {}

unsafe fn call_no_target_features() {
    target_feature_identity();
    //~^ WARNING call to `#[inline(always)]`-annotated `target_feature_identity` requires the same target features to be inlined [inline_always_mismatching_target_features]
    global_feature_enabled();
    multiple_target_features();
    //~^ WARNING call to `#[inline(always)]`-annotated `multiple_target_features` requires the same target features to be inlined [inline_always_mismatching_target_features]
}

#[target_feature(enable = "fp16,sve")]
unsafe fn call_to_first_set() {
    multiple_target_features();
    //~^ WARNING call to `#[inline(always)]`-annotated `multiple_target_features` requires the same target features to be inlined [inline_always_mismatching_target_features]
}

/* You can't have "fhm" without "fp16" */
#[target_feature(enable = "fhm")]
unsafe fn mismatching_features() {
    target_feature_identity()
}

#[target_feature(enable = "fp16")]
unsafe fn matching_target_features() {
    target_feature_identity()
}

#[inline(always)]
#[target_feature(enable = "neon")]
unsafe fn global_feature_enabled() {

}

#[inline(always)]
#[target_feature(enable = "fp16,sve")]
#[target_feature(enable="rdm")]
fn multiple_target_features() {

 }
