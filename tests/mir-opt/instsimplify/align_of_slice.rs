//@ test-mir-pass: InstSimplify-after-simplifycfg
//@ needs-unwind

#![crate_type = "lib"]
#![feature(core_intrinsics)]

// EMIT_MIR align_of_slice.of_val_slice.InstSimplify-after-simplifycfg.diff
pub fn of_val_slice<T>(slice: &[T]) -> usize {
    // CHECK-LABEL: fn of_val_slice(_1: &[T])
    // CHECK: _0 = const <T as std::mem::SizedTypeProperties>::ALIGN;
    unsafe { core::intrinsics::align_of_val(slice) }
}

// EMIT_MIR align_of_slice.of_val_str.InstSimplify-after-simplifycfg.diff
pub fn of_val_str(s: &str) -> usize {
    // CHECK-LABEL: fn of_val_str(_1: &str)
    // CHECK: _0 = const <u8 as std::mem::SizedTypeProperties>::ALIGN;
    unsafe { core::intrinsics::align_of_val(s) }
}

pub struct Wrapper<T: ?Sized>(WrapperInner<T>);
pub struct WrapperInner<T: ?Sized>((T,));

// EMIT_MIR align_of_slice.of_val_wrapper.InstSimplify-after-simplifycfg.diff
pub fn of_val_wrapper<T>(w: &Wrapper<[T]>) -> usize {
    // CHECK-LABEL: fn of_val_wrapper(_1: &Wrapper<[T]>)
    // CHECK: _0 = const <T as std::mem::SizedTypeProperties>::ALIGN;
    unsafe { core::intrinsics::align_of_val(w) }
}
