//@ add-minicore
//@ assembly-output: emit-asm
//@ min-llvm-version: 22
//
//@ revisions: XTENSA
//@ [XTENSA] compile-flags: -Copt-level=3 --target xtensa-esp32-none-elf
//@ [XTENSA] needs-llvm-components: xtensa
#![feature(no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what GCC emits.

extern crate minicore;
use minicore::*;

pub unsafe trait VaArgSafe {}

unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
unsafe impl VaArgSafe for f64 {}
unsafe impl<T> VaArgSafe for *const T {}

#[repr(C)]
struct VaListInner {
    va_stk: *const c_void,
    va_reg: *const c_void,
    va_ndx: i32,
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
    // XTENSA: l32i a8, a2, 8
    // XTENSA-NEXT: addi a8, a8, 7
    // XTENSA-NEXT: movi a9, -8
    // XTENSA-NEXT: and a8, a8, a9
    // XTENSA-NEXT: addi a10, a8, 8
    // XTENSA-NEXT: movi a9, 32
    // XTENSA-NEXT: maxu a9, a8, a9
    // XTENSA-NEXT: movi a11, 25
    // XTENSA-NEXT: or a12, a10, a10
    // XTENSA-NEXT: bltu a10, a11, .LBB0_2
    // XTENSA-NEXT: addi a12, a9, 8
    // XTENSA-NEXT: .LBB0_2:
    // XTENSA-NEXT: s32i a12, a2, 8
    // XTENSA-NEXT: bltu a10, a11, .LBB0_4
    // XTENSA-NEXT: l32i a8, a2, 0
    // XTENSA-NEXT: add a8, a8, a9
    // XTENSA-NEXT: l32i a2, a8, 0
    // XTENSA-NEXT: l32i a3, a8, 4
    // XTENSA-NEXT: retw.n
    // XTENSA-NEXT: .LBB0_4:
    // XTENSA-NEXT: l32i a9, a2, 4
    // XTENSA-NEXT: add a8, a9, a8
    // XTENSA-NEXT: l32i a2, a8, 0
    // XTENSA-NEXT: l32i a3, a8, 4
    // XTENSA-NEXT: retw.n
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // XTENSA: l32i a8, a2, 8
    // XTENSA-NEXT: addi a8, a8, 3
    // XTENSA-NEXT: movi a9, -4
    // XTENSA-NEXT: and a8, a8, a9
    // XTENSA-NEXT: addi a10, a8, 4
    // XTENSA-NEXT: movi a9, 32
    // XTENSA-NEXT: maxu a9, a8, a9
    // XTENSA-NEXT: movi a11, 25
    // XTENSA-NEXT: or a12, a10, a10
    // XTENSA-NEXT: bltu a10, a11, .LBB1_2
    // XTENSA-NEXT: addi a12, a9, 4
    // XTENSA-NEXT: .LBB1_2:
    // XTENSA-NEXT: s32i a12, a2, 8
    // XTENSA-NEXT: bltu a10, a11, .LBB1_4
    // XTENSA-NEXT: l32i a8, a2, 0
    // XTENSA-NEXT: add a8, a8, a9
    // XTENSA-NEXT: l32i a2, a8, 0
    // XTENSA-NEXT: retw.n
    // XTENSA-NEXT: .LBB1_4:
    // XTENSA-NEXT: l32i a9, a2, 4
    // XTENSA-NEXT: add a8, a9, a8
    // XTENSA-NEXT: l32i a2, a8, 0
    // XTENSA-NEXT: retw.n
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // XTENSA: l32i a8, a2, 8
    // XTENSA-NEXT: addi a8, a8, 7
    // XTENSA-NEXT: movi a9, -8
    // XTENSA-NEXT: and a8, a8, a9
    // XTENSA-NEXT: addi a10, a8, 8
    // XTENSA-NEXT: movi a9, 32
    // XTENSA-NEXT: maxu a9, a8, a9
    // XTENSA-NEXT: movi a11, 25
    // XTENSA-NEXT: or a12, a10, a10
    // XTENSA-NEXT: bltu a10, a11, .LBB2_2
    // XTENSA-NEXT: addi a12, a9, 8
    // XTENSA-NEXT: .LBB2_2:
    // XTENSA-NEXT: s32i a12, a2, 8
    // XTENSA-NEXT: bltu a10, a11, .LBB2_4
    // XTENSA-NEXT: l32i a8, a2, 0
    // XTENSA-NEXT: add a8, a8, a9
    // XTENSA-NEXT: l32i a2, a8, 0
    // XTENSA-NEXT: l32i a3, a8, 4
    // XTENSA-NEXT: retw.n
    // XTENSA-NEXT: .LBB2_4:
    // XTENSA-NEXT: l32i a9, a2, 4
    // XTENSA-NEXT: add a8, a9, a8
    // XTENSA-NEXT: l32i a2, a8, 0
    // XTENSA-NEXT: l32i a3, a8, 4
    // XTENSA-NEXT: retw.n
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // XTENSA: read_ptr = read_i32
    va_arg(ap)
}
