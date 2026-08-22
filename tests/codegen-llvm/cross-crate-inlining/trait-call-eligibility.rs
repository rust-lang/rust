//@ compile-flags: -Copt-level=3 -Zcross-crate-inline-threshold=100
//@ aux-build:trait_call_eligibility.rs

#![crate_type = "lib"]

extern crate trait_call_eligibility_aux as eligibility;

// A statically selected, explicitly inline trait implementation
// does not disqualify the enclosing body from inferred cross-crate inlining.
// CHECK-LABEL: @explicit_inline_trait_call_outer(
#[no_mangle]
pub fn explicit_inline_trait_call_outer(value: u32) -> u32 {
    // CHECK-NOT: call {{.*}}encloses_explicit_inline_trait_call
    // CHECK: add i32 %value, 10
    // CHECK-NOT: call {{.*}}encloses_explicit_inline_trait_call
    eligibility::encloses_explicit_inline_trait_call(value)
}

// A selected implementation without an inline attribute remains disqualifying.
// CHECK-LABEL: @no_inline_trait_call_outer(
#[no_mangle]
pub fn no_inline_trait_call_outer(value: u32) -> u32 {
    // CHECK: call {{.*}}encloses_no_inline_trait_call
    eligibility::encloses_no_inline_trait_call(value)
}

// A remaining direct call to an explicitly inline non-trait function
// still prevents inferred cross-crate inlining.
// CHECK-LABEL: @direct_inline_call_outer(
#[no_mangle]
pub fn direct_inline_call_outer(value: u32) -> u32 {
    // CHECK: call {{.*}}encloses_direct_inline_call
    eligibility::encloses_direct_inline_call(value)
}

// A virtual selection is not a statically selected item and is disqualifying.
// CHECK-LABEL: @inline_virtual_call_outer(
#[no_mangle]
pub fn inline_virtual_call_outer(callee: &dyn eligibility::InlineVirtualCall, value: u32) -> u32 {
    // CHECK: call {{.*}}encloses_inline_virtual_call
    eligibility::encloses_inline_virtual_call(callee, value)
}
