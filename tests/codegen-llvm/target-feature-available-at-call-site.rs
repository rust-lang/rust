//@ compile-flags: --crate-type=lib -Copt-level=3 -Ctarget-feature=-avx
//@ only-x86_64
//@ ignore-backends: gcc

#![feature(core_intrinsics)]

use std::intrinsics::simd::target_feature_available_at_call_site;

#[inline]
pub fn avx_branch_value() -> i32 {
    if target_feature_available_at_call_site!("avx") { 1 } else { 0 }
}

// CHECK-LABEL: @with_avx(
// CHECK-NOT: rust.target_feature_available_at_call_site
// CHECK: ret i32 1
#[no_mangle]
#[target_feature(enable = "avx")]
pub fn with_avx() -> i32 {
    avx_branch_value()
}

// CHECK-LABEL: @without_avx(
// CHECK-NOT: rust.target_feature_available_at_call_site
// CHECK: ret i32 0
#[no_mangle]
pub fn without_avx() -> i32 {
    avx_branch_value()
}

// CHECK: attributes #0 = {{.*}}"target-features"="{{[^"]*}}+avx{{.*}}"
// CHECK: attributes #1 = {{.*}}"target-features"="{{[^"]*}}-avx{{.*}}"
