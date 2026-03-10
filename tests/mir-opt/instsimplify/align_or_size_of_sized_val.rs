//@ test-mir-pass: InstSimplify-after-simplifycfg
//@ needs-unwind

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// EMIT_MIR align_or_size_of_sized_val.align_of_val_sized.InstSimplify-after-simplifycfg.diff
pub fn align_of_val_sized<T>(val: &T) -> usize {
    // CHECK-LABEL: fn align_of_val_sized
    // CHECK: _0 = const <T as std::mem::SizedTypeProperties>::ALIGN;
    unsafe { core::intrinsics::align_of_val(val) }
}

// EMIT_MIR align_or_size_of_sized_val.size_of_val_sized.InstSimplify-after-simplifycfg.diff
pub fn size_of_val_sized<T>(val: &T) -> usize {
    // CHECK-LABEL: fn size_of_val_sized
    // CHECK: _0 = const <T as std::mem::SizedTypeProperties>::SIZE;
    unsafe { core::intrinsics::size_of_val(val) }
}
