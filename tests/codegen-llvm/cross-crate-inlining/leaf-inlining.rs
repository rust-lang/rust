//@ compile-flags: -Copt-level=3 -Zcross-crate-inline-threshold=yes
//@ aux-build:leaf.rs

#![crate_type = "lib"]

extern crate leaf;

// Check that we inline a leaf cross-crate call
// CHECK-LABEL: @leaf_outer(
#[no_mangle]
pub fn leaf_outer() -> String {
    // CHECK-NOT: call {{.*}}leaf_fn
    leaf::leaf_fn()
}

// Check that we do not inline a non-leaf cross-crate call
// CHECK-LABEL: @stem_outer(
#[no_mangle]
pub fn stem_outer() -> String {
    // CHECK: call {{.*}}stem_fn
    leaf::stem_fn()
}

// Check that we inline functions that call intrinsics
// CHECK-LABEL: @leaf_with_intrinsic_outer(
#[no_mangle]
pub fn leaf_with_intrinsic_outer(a: &[u64; 2], b: &[u64; 2]) -> bool {
    // CHECK-NOT: call {{.*}}leaf_with_intrinsic
    leaf::leaf_with_intrinsic(a, b)
}

// Check that we inline functions with assert terminators
// CHECK-LABEL: @leaf_with_assert(
#[no_mangle]
pub fn leaf_with_assert(a: i32, b: i32) -> i32 {
    // CHECK-NOT: call {{.*}}leaf_with_assert
    // CHECK: sdiv i32 %a, %b
    // CHECK-NOT: call {{.*}}leaf_with_assert
    leaf::leaf_with_assert(a, b)
}

// Check that we inline functions with only statically dispatched trait calls.
// The selected implementations are inline.
// CHECK-LABEL: @leaf_with_inline_trait_calls(
#[no_mangle]
pub fn leaf_with_inline_trait_calls(input: &str, token: char) -> bool {
    // CHECK-NOT: call {{.*}}find_token
    <&str as leaf::FindToken<char>>::find_token(&input, token)
}

// Check that we do not inline functions with statically dispatched trait calls
// when the selected implementations are not inline.
// CHECK-LABEL: @stem_with_non_inline_trait_call(
#[no_mangle]
pub fn stem_with_non_inline_trait_call(value: u32) -> u32 {
    // CHECK: call {{.*}}stem_with_non_inline_trait_call
    leaf::stem_with_non_inline_trait_call(value)
}

// Check that we do not inline functions with statically dispatched trait calls
// when the selected implementations are exported and therefore globally shared.
// CHECK-LABEL: @stem_with_exported_inline_trait_call(
#[no_mangle]
pub fn stem_with_exported_inline_trait_call(value: u32) -> u32 {
    // CHECK: call {{.*}}stem_with_exported_inline_trait_call
    leaf::stem_with_exported_inline_trait_call(value)
}
