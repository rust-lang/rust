//@ test-mir-pass: InstSimplify-after-simplifycfg
//@ needs-unwind

#![crate_type = "lib"]
#![feature(core_intrinsics)]
#![feature(ptr_alignment_type)]

// EMIT_MIR align_of_slice.of_val_slice.InstSimplify-after-simplifycfg.diff
pub fn of_val_slice<T>(slice: &[T]) -> std::ptr::Alignment {
    // CHECK-LABEL: fn of_val_slice(_1: &[T])
    // CHECK: _0 = const <T as std::mem::SizedTypeProperties>::ALIGNMENT;
    unsafe { core::intrinsics::align_of_val(slice) }
}
