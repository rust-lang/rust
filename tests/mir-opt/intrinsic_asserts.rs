#![crate_type = "lib"]
#![feature(core_intrinsics)]

// All these assertions pass, so all the intrinsic calls should be deleted.
// EMIT_MIR intrinsic_asserts.removable.InstSimplify.diff
pub fn removable() {
    core::intrinsics::assert_inhabited::<()>();
    core::intrinsics::assert_zero_valid::<u8>();
    core::intrinsics::assert_mem_uninitialized_valid::<u8>();
}

enum Never {}

// These assertions all diverge, so their target blocks should become None.
// EMIT_MIR intrinsic_asserts.panics.InstSimplify.diff
pub fn panics() {
    core::intrinsics::assert_inhabited::<Never>();
    core::intrinsics::assert_zero_valid::<&u8>();
    core::intrinsics::assert_mem_uninitialized_valid::<&u8>();
}

// Whether or not these asserts pass isn't known, so they shouldn't be modified.
// EMIT_MIR intrinsic_asserts.generic.InstSimplify.diff
pub fn generic<T>() {
    core::intrinsics::assert_inhabited::<T>();
    core::intrinsics::assert_zero_valid::<T>();
    core::intrinsics::assert_mem_uninitialized_valid::<T>();
}
