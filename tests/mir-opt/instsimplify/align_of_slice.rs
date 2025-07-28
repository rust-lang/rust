//@ test-mir-pass: InstSimplify-after-simplifycfg
//@ needs-unwind

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// EMIT_MIR align_of_slice.of_val_slice.InstSimplify-after-simplifycfg.diff
pub fn of_val_slice<T>(slice: &[T]) -> usize {
    // CHECK-LABEL: fn of_val_slice(_1: &[T])
    // CHECK: _2 = &raw const (*_1);
    // CHECK: _0 = std::intrinsics::align_of_val::<[T]>(move _2)
    unsafe { core::intrinsics::align_of_val(slice) }
}
