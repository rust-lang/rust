//@ test-mir-pass: InstSimplify-after-simplifycfg

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// All these assertions pass, so all the intrinsic calls should be deleted.
// EMIT_MIR intrinsic_asserts.removable.InstSimplify-after-simplifycfg.diff
pub fn removable() {
    // CHECK-LABEL: fn removable(
    // CHECK-NOT: assert_inhabited
    // CHECK-NOT: assert_zero_valid
    // CHECK-NOT: assert_mem_uninitialized_valid
    core::intrinsics::assert_inhabited::<()>();
    core::intrinsics::assert_zero_valid::<u8>();
    core::intrinsics::assert_mem_uninitialized_valid::<u8>();
}

enum Never {}

// These assertions all diverge, so their target blocks should become None.
// EMIT_MIR intrinsic_asserts.panics.InstSimplify-after-simplifycfg.diff
pub fn panics() {
    // CHECK-LABEL: fn panics(
    // CHECK: assert_inhabited::<Never>() -> unwind
    // CHECK: assert_zero_valid::<&u8>() -> unwind
    // CHECK: assert_mem_uninitialized_valid::<&u8>() -> unwind
    core::intrinsics::assert_inhabited::<Never>();
    core::intrinsics::assert_zero_valid::<&u8>();
    core::intrinsics::assert_mem_uninitialized_valid::<&u8>();
}

// Whether or not these asserts pass isn't known, so they shouldn't be modified.
// EMIT_MIR intrinsic_asserts.generic.InstSimplify-after-simplifycfg.diff
pub fn generic<T>() {
    // CHECK-LABEL: fn generic(
    // CHECK: assert_inhabited::<T>() -> [return:
    // CHECK: assert_zero_valid::<T>() -> [return:
    // CHECK: assert_mem_uninitialized_valid::<T>() -> [return:
    core::intrinsics::assert_inhabited::<T>();
    core::intrinsics::assert_zero_valid::<T>();
    core::intrinsics::assert_mem_uninitialized_valid::<T>();
}

// Whether or not these asserts pass isn't known, so they shouldn't be modified.
// EMIT_MIR intrinsic_asserts.generic_ref.InstSimplify-after-simplifycfg.diff
pub fn generic_ref<T>() {
    // CHECK-LABEL: fn generic_ref(
    // CHECK: assert_mem_uninitialized_valid::<&T>() -> [return:
    core::intrinsics::assert_mem_uninitialized_valid::<&T>();
}
