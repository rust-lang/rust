//@ compile-flags: --crate-type=lib -Copt-level=3
//@ only-aarch64
//@ ignore-backends: gcc

// Test intrinsic on a feature that doesn't map to an LLVM feature.
// Can't resolve after inlining, but can get the current function's
// feature presence.

#![feature(core_intrinsics)]

use std::intrinsics::simd::target_feature_available_at_call_site;

// CHECK-LABEL: @with_tme(
// CHECK-NOT: rust.target_feature_available_at_call_site
// CHECK: ret i1 true
#[no_mangle]
#[target_feature(enable = "tme")]
pub fn with_tme() -> bool {
    target_feature_available_at_call_site!("tme")
}

// CHECK-LABEL: @without_tme(
// CHECK-NOT: rust.target_feature_available_at_call_site
// CHECK: ret i1 false
#[no_mangle]
pub fn without_tme() -> bool {
    target_feature_available_at_call_site!("tme")
}
