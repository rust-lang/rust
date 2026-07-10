//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: RISCV32 RISCV64
//@ [RISCV32] compile-flags: -Copt-level=3 --target riscv32gc-unknown-linux-gnu
//@ [RISCV32] needs-llvm-components: riscv
//@ [RISCV64] compile-flags: -Copt-level=3 --target riscv64gc-unknown-linux-gnu
//@ [RISCV64] needs-llvm-components: riscv
#![feature(c_variadic, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

extern crate minicore;
use minicore::*;

#[lang = "va_arg_safe"]
pub unsafe trait VaArgSafe {}

unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
#[cfg(target_pointer_width = "64")]
unsafe impl VaArgSafe for i128 {}
unsafe impl VaArgSafe for f64 {}
unsafe impl<T> VaArgSafe for *const T {}

#[repr(transparent)]
struct VaListInner {
    ptr: *const c_void,
}

#[repr(transparent)]
#[lang = "va_list"]
pub struct VaList<'a> {
    inner: VaListInner,
    _marker: PhantomData<&'a mut ()>,
}

#[rustc_intrinsic]
#[rustc_nounwind]
pub const unsafe fn va_arg<T: VaArgSafe>(ap: &mut VaList<'_>) -> T;

#[unsafe(no_mangle)]
unsafe extern "C" fn read_f64(ap: &mut VaList<'_>) -> f64 {
    // CHECK-LABEL: read_f64
    //
    // RISCV32: lw a1, 0(a0)
    // RISCV32-NEXT: addi a1, a1, 7
    // RISCV32-NEXT: andi a1, a1, -8
    // RISCV32-NEXT: fld fa0, 0(a1)
    // RISCV32-NEXT: addi a1, a1, 8
    // RISCV32-NEXT: sw a1, 0(a0)
    // RISCV32-NEXT: ret
    //
    // RISCV64: ld a1, 0(a0)
    // RISCV64-NEXT: fld fa0, 0(a1)
    // RISCV64-NEXT: addi a1, a1, 8
    // RISCV64-NEXT: sd a1, 0(a0)
    // RISCV64-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // RISCV32: lw a2, 0(a0)
    // RISCV32-NEXT: lw a1, 0(a2)
    // RISCV32-NEXT: addi a2, a2, 4
    // RISCV32-NEXT: sw a2, 0(a0)
    // RISCV32-NEXT: mv a0, a1
    // RISCV32-NEXT: ret
    //
    // RISCV64: ld a2, 0(a0)
    // RISCV64-NEXT: lw a1, 0(a2)
    // RISCV64-NEXT: addi a2, a2, 8
    // RISCV64-NEXT: sd a2, 0(a0)
    // RISCV64-NEXT: mv a0, a1
    // RISCV64-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // RISCV32: lw a1, 0(a0)
    // RISCV32-NEXT: addi a1, a1, 7
    // RISCV32-NEXT: andi a3, a1, -8
    // RISCV32-NEXT: lw a2, 0(a3)
    // RISCV32-NEXT: lw a1, 4(a3)
    // RISCV32-NEXT: addi a3, a3, 8
    // RISCV32-NEXT: sw a3, 0(a0)
    // RISCV32-NEXT: mv a0, a2
    // RISCV32-NEXT: ret
    //
    // RISCV64: ld a2, 0(a0)
    // RISCV64-NEXT: ld a1, 0(a2)
    // RISCV64-NEXT: addi a2, a2, 8
    // RISCV64-NEXT: sd a2, 0(a0)
    // RISCV64-NEXT: mv a0, a1
    // RISCV64-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
#[cfg(target_pointer_width = "64")]
unsafe extern "C" fn read_i128(ap: &mut VaList<'_>) -> i128 {
    // RISCV64-LABEL: read_i128
    //
    // RISCV64: ld a1, 0(a0)
    // RISCV64-NEXT: addi a1, a1, 15
    // RISCV64-NEXT: andi a3, a1, -16
    // RISCV64-NEXT: ld a2, 0(a3)
    // RISCV64-NEXT: ld a1, 8(a3)
    // RISCV64-NEXT: addi a3, a3, 16
    // RISCV64-NEXT: sd a3, 0(a0)
    // RISCV64-NEXT: mv a0, a2
    // RISCV64-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // RISCV32: read_ptr = read_i32
    // RISCV64: read_ptr = read_i64
    va_arg(ap)
}
