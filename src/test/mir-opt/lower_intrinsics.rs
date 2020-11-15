// compile-flags: -Cpanic=abort
#![feature(core_intrinsics)]
#![crate_type = "lib"]

// EMIT_MIR lower_intrinsics.wrapping.LowerIntrinsics.diff
pub fn wrapping<T: Copy>(a: T, b: T) {
    let _x = core::intrinsics::wrapping_add(a, b);
    let _y = core::intrinsics::wrapping_sub(a, b);
    let _z = core::intrinsics::wrapping_mul(a, b);
}

// EMIT_MIR lower_intrinsics.size_of.LowerIntrinsics.diff
pub fn size_of<T>() -> usize {
    core::intrinsics::size_of::<T>()
}

// EMIT_MIR lower_intrinsics.forget.LowerIntrinsics.diff
pub fn forget<T>(t: T) {
    unsafe { core::intrinsics::forget(t) };
}

// EMIT_MIR lower_intrinsics.unreachable.LowerIntrinsics.diff
pub fn unreachable() -> ! {
    unsafe { core::intrinsics::unreachable() };
}

// EMIT_MIR lower_intrinsics.f_unit.PreCodegen.before.mir
pub fn f_unit() {
    f_dispatch(());
}


// EMIT_MIR lower_intrinsics.f_u64.PreCodegen.before.mir
pub fn f_u64() {
    f_dispatch(0u64);
}

#[inline(always)]
pub fn f_dispatch<T>(t: T) {
    if std::mem::size_of::<T>() == 0 {
        f_zst(t);
    } else {
        f_non_zst(t);
    }
}

#[inline(never)]
pub fn f_zst<T>(t: T) {
}

#[inline(never)]
pub fn f_non_zst<T>(t: T) {}
