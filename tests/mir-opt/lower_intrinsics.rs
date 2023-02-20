// unit-test: LowerIntrinsics
// ignore-wasm32 compiled with panic=abort by default

#![feature(core_intrinsics, intrinsics)]
#![crate_type = "lib"]

// EMIT_MIR lower_intrinsics.wrapping.LowerIntrinsics.diff
pub fn wrapping(a: i32, b: i32) {
    let _x = core::intrinsics::wrapping_add(a, b);
    let _y = core::intrinsics::wrapping_sub(a, b);
    let _z = core::intrinsics::wrapping_mul(a, b);
}

// EMIT_MIR lower_intrinsics.size_of.LowerIntrinsics.diff
pub fn size_of<T>() -> usize {
    core::intrinsics::size_of::<T>()
}

// EMIT_MIR lower_intrinsics.align_of.LowerIntrinsics.diff
pub fn align_of<T>() -> usize {
    core::intrinsics::min_align_of::<T>()
}

// EMIT_MIR lower_intrinsics.forget.LowerIntrinsics.diff
pub fn forget<T>(t: T) {
    core::intrinsics::forget(t)
}

// EMIT_MIR lower_intrinsics.unreachable.LowerIntrinsics.diff
pub fn unreachable() -> ! {
    unsafe { core::intrinsics::unreachable() };
}

// EMIT_MIR lower_intrinsics.non_const.LowerIntrinsics.diff
pub fn non_const<T>() -> usize {
    // Check that lowering works with non-const operand as a func.
    let size_of_t = core::intrinsics::size_of::<T>;
    size_of_t()
}

pub enum E {
    A,
    B,
    C,
}

// EMIT_MIR lower_intrinsics.discriminant.LowerIntrinsics.diff
pub fn discriminant<T>(t: T) {
    core::intrinsics::discriminant_value(&t);
    core::intrinsics::discriminant_value(&0);
    core::intrinsics::discriminant_value(&());
    core::intrinsics::discriminant_value(&E::B);
}

extern "rust-intrinsic" {
    // Cannot use `std::intrinsics::copy_nonoverlapping` as that is a wrapper function
    fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);
}

// EMIT_MIR lower_intrinsics.f_copy_nonoverlapping.LowerIntrinsics.diff
pub fn f_copy_nonoverlapping() {
    let src = ();
    let mut dst = ();
    unsafe {
        copy_nonoverlapping(&src as *const _ as *const i32, &mut dst as *mut _ as *mut i32, 0);
    }
}

// EMIT_MIR lower_intrinsics.assume.LowerIntrinsics.diff
pub fn assume() {
    unsafe {
        std::intrinsics::assume(true);
    }
}

// EMIT_MIR lower_intrinsics.with_overflow.LowerIntrinsics.diff
pub fn with_overflow(a: i32, b: i32) {
    let _x = core::intrinsics::add_with_overflow(a, b);
    let _y = core::intrinsics::sub_with_overflow(a, b);
    let _z = core::intrinsics::mul_with_overflow(a, b);
}
