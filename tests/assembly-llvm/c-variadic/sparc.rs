//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: SPARC SPARC64
//@ [SPARC] compile-flags: -Copt-level=3 --target sparc-unknown-linux-gnu
//@ [SPARC] needs-llvm-components: sparc
//@ [SPARC64] compile-flags: -Copt-level=3 --target sparc64-unknown-linux-gnu
//@ [SPARC64] needs-llvm-components: sparc
#![feature(no_core, lang_items, intrinsics, rustc_attrs, asm_experimental_arch)]
#![no_core]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what clang emits.

extern crate minicore;
use minicore::*;

#[lang = "va_arg_safe"]
pub unsafe trait VaArgSafe {}

unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
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
    // SPARC: ld [%o0], %o1
    // SPARC-NEXT: add %o1, 8, %o2
    // SPARC-NEXT: st %o2, [%o0]
    // SPARC-NEXT: ld [%o1+4], %o0
    // SPARC-NEXT: add %sp, 96, %o2
    // SPARC-NEXT: or %o2, 4, %o2
    // SPARC-NEXT: st %o0, [%o2]
    // SPARC-NEXT: ld [%o1], %o0
    // SPARC-NEXT: st %o0, [%sp+96]
    // SPARC-NEXT: ldd [%sp+96], %f0
    // SPARC-NEXT: retl
    // SPARC-NEXT: add %sp, 104, %sp
    //
    // SPARC64: ldx [%o0], %o1
    // SPARC64-NEXT: add %o1, 8, %o2
    // SPARC64-NEXT: stx %o2, [%o0]
    // SPARC64-NEXT: retl
    // SPARC64-NEXT: ldd [%o1], %f0
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // SPARC: ld [%o0], %o1
    // SPARC-NEXT: add %o1, 4, %o2
    // SPARC-NEXT: st %o2, [%o0]
    // SPARC-NEXT: retl
    // SPARC-NEXT: ld [%o1], %o0
    //
    // SPARC64: ldx [%o0], %o1
    // SPARC64-NEXT: add %o1, 8, %o2
    // SPARC64-NEXT: stx %o2, [%o0]
    // SPARC64-NEXT: retl
    // SPARC64-NEXT: ldsw [%o1+4], %o0
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64
    //
    // SPARC: ld [%o0], %o1
    // SPARC-NEXT: add %o1, 4, %o2
    // SPARC-NEXT: st %o2, [%o0]
    // SPARC-NEXT: ld [%o1], %o2
    // SPARC-NEXT: add %o1, 8, %o3
    // SPARC-NEXT: st %o3, [%o0]
    // SPARC-NEXT: ld [%o1+4], %o1
    // SPARC-NEXT: retl
    // SPARC-NEXT: mov %o2, %o0
    //
    // SPARC64: ldx [%o0], %o1
    // SPARC64-NEXT: add %o1, 8, %o2
    // SPARC64-NEXT: stx %o2, [%o0]
    // SPARC64-NEXT: retl
    // SPARC64-NEXT: ldx [%o1], %o0
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // SPARC: read_ptr = read_i32
    // SPARC64: read_ptr = read_i64
    va_arg(ap)
}
