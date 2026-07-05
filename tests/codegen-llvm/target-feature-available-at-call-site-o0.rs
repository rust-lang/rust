//@ revisions: NEG_ONLY NEG_POS POS_NEG
//@ compile-flags: --crate-type=lib -Copt-level=0
//@ [NEG_ONLY] compile-flags: -Ctarget-feature=-fma
//@ [NEG_POS] compile-flags: -Ctarget-feature=-fma,+fma
//@ [POS_NEG] compile-flags: -Ctarget-feature=+fma,-fma
//@ only-x86_64
//@ ignore-backends: gcc

// opt-level=0 has different inlining properties, so ensure that the pass still works.

#![feature(core_intrinsics)]

use std::intrinsics::simd::target_feature_available_at_call_site;

// NEG_ONLY-LABEL: @with_fma(
// NEG_ONLY-NOT: rust.target_feature_available_at_call_site
// NEG_ONLY: ret i1 true
// NEG_POS-LABEL: @with_fma(
// NEG_POS-NOT: rust.target_feature_available_at_call_site
// NEG_POS: ret i1 true
// POS_NEG-LABEL: @with_fma(
// POS_NEG-NOT: rust.target_feature_available_at_call_site
// POS_NEG: ret i1 true
#[no_mangle]
#[target_feature(enable = "fma")]
pub fn with_fma() -> bool {
    target_feature_available_at_call_site!("fma")
}

// NEG_ONLY-LABEL: @without_fma(
// NEG_ONLY-NOT: rust.target_feature_available_at_call_site
// NEG_ONLY: ret i1 false
// NEG_POS-LABEL: @without_fma(
// NEG_POS-NOT: rust.target_feature_available_at_call_site
// NEG_POS: ret i1 true
// POS_NEG-LABEL: @without_fma(
// POS_NEG-NOT: rust.target_feature_available_at_call_site
// POS_NEG: ret i1 false
#[no_mangle]
pub fn without_fma() -> bool {
    target_feature_available_at_call_site!("fma")
}
