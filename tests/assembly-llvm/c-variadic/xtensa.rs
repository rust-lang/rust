//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: XTENSA
//@ [XTENSA] compile-flags: -Copt-level=3 --target xtensa-esp32-none-elf
//@ [XTENSA] needs-llvm-components: xtensa
#![feature(c_variadic, no_core, lang_items, intrinsics, rustc_attrs)]
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
    // XTENSA: l32i{{(\.n)?}} [[F64_NDX:a[0-9]+]], a2, 8
    // XTENSA: addi{{(\.n)?}} [[F64_ALIGNED_TMP:a[0-9]+]], [[F64_NDX]], 7
    // XTENSA: movi{{(\.n)?}} [[F64_ALIGN_MASK:a[0-9]+]], -8
    // XTENSA: and [[F64_OFF:a[0-9]+]], [[F64_ALIGNED_TMP]], [[F64_ALIGN_MASK]]
    // XTENSA: addi{{(\.n)?}} [[F64_NEXT:a[0-9]+]], [[F64_OFF]], 8
    // XTENSA: maxu [[F64_STACK_OFF:a[0-9]+]], [[F64_OFF]], {{a[0-9]+}}
    // XTENSA: bltu [[F64_NEXT]], {{a[0-9]+}}, [[F64_REGSAVE:.LBB[0-9]+_[0-9]+]]
    // XTENSA: addi{{(\.n)?}} [[F64_UPDATED_NDX:a[0-9]+]], [[F64_STACK_OFF]], 8
    // XTENSA: [[F64_UPDATE:.LBB[0-9]+_[0-9]+]]:
    // XTENSA-NEXT: s32i{{(\.n)?}} {{a[0-9]+}}, a2, 8
    // XTENSA: bltu [[F64_NEXT]], {{a[0-9]+}}, [[F64_REGSAVE_LOAD:.LBB[0-9]+_[0-9]+]]
    // XTENSA: l32i{{(\.n)?}} [[F64_STACK:a[0-9]+]], a2, 0
    // XTENSA: add{{(\.n)?}} [[F64_PTR:a[0-9]+]], [[F64_STACK]], [[F64_STACK_OFF]]
    // XTENSA: l32i{{(\.n)?}} a2, [[F64_PTR]], 0
    // XTENSA-NEXT: l32i{{(\.n)?}} a3, [[F64_PTR]], 4
    // XTENSA-NEXT: retw.n
    // XTENSA: [[F64_REGSAVE_LOAD]]:
    // XTENSA-NEXT: l32i{{(\.n)?}} [[F64_REGSAVE_AREA:a[0-9]+]], a2, 4
    // XTENSA-NEXT: add{{(\.n)?}} [[F64_REGSAVE_PTR:a[0-9]+]], [[F64_REGSAVE_AREA]], [[F64_OFF]]
    // XTENSA-NEXT: l32i{{(\.n)?}} a2, [[F64_REGSAVE_PTR]], 0
    // XTENSA-NEXT: l32i{{(\.n)?}} a3, [[F64_REGSAVE_PTR]], 4
    // XTENSA-NEXT: retw.n
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // XTENSA: l32i{{(\.n)?}} [[I32_NDX:a[0-9]+]], a2, 8
    // XTENSA: addi{{(\.n)?}} [[I32_ALIGNED_TMP:a[0-9]+]], [[I32_NDX]], 3
    // XTENSA: movi{{(\.n)?}} [[I32_ALIGN_MASK:a[0-9]+]], -4
    // XTENSA: and [[I32_OFF:a[0-9]+]], [[I32_ALIGNED_TMP]], [[I32_ALIGN_MASK]]
    // XTENSA: addi{{(\.n)?}} [[I32_NEXT:a[0-9]+]], [[I32_OFF]], 4
    // XTENSA: maxu [[I32_STACK_OFF:a[0-9]+]], [[I32_OFF]], {{a[0-9]+}}
    // XTENSA: bltu [[I32_NEXT]], {{a[0-9]+}}, [[I32_REGSAVE:.LBB[0-9]+_[0-9]+]]
    // XTENSA: addi{{(\.n)?}} [[I32_UPDATED_NDX:a[0-9]+]], [[I32_STACK_OFF]], 4
    // XTENSA: [[I32_UPDATE:.LBB[0-9]+_[0-9]+]]:
    // XTENSA-NEXT: s32i{{(\.n)?}} {{a[0-9]+}}, a2, 8
    // XTENSA: bltu [[I32_NEXT]], {{a[0-9]+}}, [[I32_REGSAVE_LOAD:.LBB[0-9]+_[0-9]+]]
    // XTENSA: l32i{{(\.n)?}} [[I32_STACK:a[0-9]+]], a2, 0
    // XTENSA: add{{(\.n)?}} [[I32_PTR:a[0-9]+]], [[I32_STACK]], [[I32_STACK_OFF]]
    // XTENSA: l32i{{(\.n)?}} a2, [[I32_PTR]], 0
    // XTENSA-NEXT: retw.n
    // XTENSA: [[I32_REGSAVE_LOAD]]:
    // XTENSA-NEXT: l32i{{(\.n)?}} [[I32_REGSAVE_AREA:a[0-9]+]], a2, 4
    // XTENSA-NEXT: add{{(\.n)?}} [[I32_REGSAVE_PTR:a[0-9]+]], [[I32_REGSAVE_AREA]], [[I32_OFF]]
    // XTENSA-NEXT: l32i{{(\.n)?}} a2, [[I32_REGSAVE_PTR]], 0
    // XTENSA-NEXT: retw.n
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // XTENSA: l32i{{(\.n)?}} [[I64_NDX:a[0-9]+]], a2, 8
    // XTENSA: addi{{(\.n)?}} [[I64_ALIGNED_TMP:a[0-9]+]], [[I64_NDX]], 7
    // XTENSA: movi{{(\.n)?}} [[I64_ALIGN_MASK:a[0-9]+]], -8
    // XTENSA: and [[I64_OFF:a[0-9]+]], [[I64_ALIGNED_TMP]], [[I64_ALIGN_MASK]]
    // XTENSA: addi{{(\.n)?}} [[I64_NEXT:a[0-9]+]], [[I64_OFF]], 8
    // XTENSA: maxu [[I64_STACK_OFF:a[0-9]+]], [[I64_OFF]], {{a[0-9]+}}
    // XTENSA: bltu [[I64_NEXT]], {{a[0-9]+}}, [[I64_REGSAVE:.LBB[0-9]+_[0-9]+]]
    // XTENSA: addi{{(\.n)?}} [[I64_UPDATED_NDX:a[0-9]+]], [[I64_STACK_OFF]], 8
    // XTENSA: [[I64_UPDATE:.LBB[0-9]+_[0-9]+]]:
    // XTENSA-NEXT: s32i{{(\.n)?}} {{a[0-9]+}}, a2, 8
    // XTENSA: bltu [[I64_NEXT]], {{a[0-9]+}}, [[I64_REGSAVE_LOAD:.LBB[0-9]+_[0-9]+]]
    // XTENSA: l32i{{(\.n)?}} [[I64_STACK:a[0-9]+]], a2, 0
    // XTENSA: add{{(\.n)?}} [[I64_PTR:a[0-9]+]], [[I64_STACK]], [[I64_STACK_OFF]]
    // XTENSA: l32i{{(\.n)?}} a2, [[I64_PTR]], 0
    // XTENSA-NEXT: l32i{{(\.n)?}} a3, [[I64_PTR]], 4
    // XTENSA-NEXT: retw.n
    // XTENSA: [[I64_REGSAVE_LOAD]]:
    // XTENSA-NEXT: l32i{{(\.n)?}} [[I64_REGSAVE_AREA:a[0-9]+]], a2, 4
    // XTENSA-NEXT: add{{(\.n)?}} [[I64_REGSAVE_PTR:a[0-9]+]], [[I64_REGSAVE_AREA]], [[I64_OFF]]
    // XTENSA-NEXT: l32i{{(\.n)?}} a2, [[I64_REGSAVE_PTR]], 0
    // XTENSA-NEXT: l32i{{(\.n)?}} a3, [[I64_REGSAVE_PTR]], 4
    // XTENSA-NEXT: retw.n
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // XTENSA: read_ptr = read_i32
    va_arg(ap)
}
