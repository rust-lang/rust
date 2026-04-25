//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: AARCH64_LINUX AARCH64_DARWIN AARCH64_BE ARM64EC_MSVC
//@ [AARCH64_LINUX] compile-flags: -Copt-level=3 --target aarch64-unknown-linux-gnu
//@ [AARCH64_LINUX] needs-llvm-components: aarch64
//@ [AARCH64_BE] compile-flags: -Copt-level=3 --target aarch64_be-unknown-linux-gnu
//@ [AARCH64_BE] needs-llvm-components: aarch64
//@ [AARCH64_DARWIN] compile-flags: -Copt-level=3 --target aarch64-apple-darwin
//@ [AARCH64_DARWIN] needs-llvm-components: aarch64
//@ [ARM64EC_MSVC] compile-flags: -Copt-level=3 --target arm64ec-pc-windows-msvc
//@ [ARM64EC_MSVC] needs-llvm-components: aarch64
#![feature(c_variadic, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what clang emits.

// For aarch64-unknown-linux-gnu LLVM canonicalizes a comparison, leading to slightly different
// assembly.
//
// For aarch64-apple-darwin LLVM is able to optimize our output better, because we effectively
// desugar va_arg early, hence we don't actually match Clang there.

extern crate minicore;
use minicore::*;

#[lang = "va_arg_safe"]
pub unsafe trait VaArgSafe {}

unsafe impl VaArgSafe for i32 {}
unsafe impl VaArgSafe for i64 {}
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
    // AARCH64_LINUX-LABEL: read_f64:
    // AARCH64_LINUX: ldrsw   x8, [x0, #28]
    // AARCH64_LINUX-NEXT: tbz w8, #31, .LBB0_2
    // AARCH64_LINUX-NEXT: add w9, w8, #16
    // AARCH64_LINUX-NEXT: cmn w8, #16
    // AARCH64_LINUX-NEXT: str w9, [x0, #28]
    // AARCH64_LINUX-NEXT: b.ls    .LBB0_3
    // AARCH64_LINUX-NEXT: .LBB0_2:
    // AARCH64_LINUX-NEXT: ldr x8, [x0]
    // AARCH64_LINUX-NEXT: ldr d0, [x8]
    // AARCH64_LINUX-NEXT: add x9, x8, #8
    // AARCH64_LINUX-NEXT: str x9, [x0]
    // AARCH64_LINUX-NEXT: ret
    // AARCH64_LINUX-NEXT: .LBB0_3
    // AARCH64_LINUX-NEXT: ldr x9, [x0, #16]
    // AARCH64_LINUX-NEXT: add x8, x9, x8
    // AARCH64_LINUX-NEXT: ldr d0, [x8]
    // AARCH64_LINUX-NEXT: ret

    // AARCH64_BE-LABEL: read_f64:
    // AARCH64_BE: ldrsw    x8, [x0, #28]
    // AARCH64_BE-NEXT: tbz w8, #31, .LBB0_2
    // AARCH64_BE-NEXT: add w9, w8, #16
    // AARCH64_BE-NEXT: cmn w8, #16
    // AARCH64_BE-NEXT: str w9, [x0, #28]
    // AARCH64_BE-NEXT: b.ls    .LBB0_3
    // AARCH64_BE-NEXT: .LBB0_2:
    // AARCH64_BE-NEXT: ldr x8, [x0]
    // AARCH64_BE-NEXT: ldr d0, [x8]
    // AARCH64_BE-NEXT: add x9, x8, #8
    // AARCH64_BE-NEXT: str x9, [x0]
    // AARCH64_BE-NEXT: ret
    // AARCH64_BE-NEXT: .LBB0_3:
    // AARCH64_BE-NEXT: ldr x9, [x0, #16]
    // AARCH64_BE-NEXT: add x8, x9, x8
    // AARCH64_BE-NEXT: ldr d0, [x8, #8]!
    // AARCH64_BE-NEXT: ret

    // ARM64EC_MSVC-LABEL: read_f64 = "#read_f64"
    // ARM64EC_MSVC: ldr x8, [x0]
    // ARM64EC_MSVC-NEXT: ldr d0, [x8], #8
    // ARM64EC_MSVC-NEXT: str x8, [x0]
    // ARM64EC_MSVC-NEXT: ret

    // AARCH64_DARWIN-LABEL: _read_f64:
    // AARCH64_DARWIN: ldr x8, [x0]
    // AARCH64_DARWIN-NEXT: ldr d0, [x8], #8
    // AARCH64_DARWIN-NEXT: str x8, [x0]
    // AARCH64_DARWIN-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // AARCH64_LINUX-LABEL: read_i32:
    // AARCH64_LINUX: ldrsw   x8, [x0, #24]
    // AARCH64_LINUX-NEXT: tbz w8, #31, .LBB1_2
    // AARCH64_LINUX-NEXT: add w9, w8, #8
    // AARCH64_LINUX-NEXT: cmn w8, #8
    // AARCH64_LINUX-NEXT: str w9, [x0, #24]
    // AARCH64_LINUX-NEXT: b.ls    .LBB1_3
    // AARCH64_LINUX-NEXT: .LBB1_2:
    // AARCH64_LINUX-NEXT: ldr x8, [x0]
    // AARCH64_LINUX-NEXT: add x9, x8, #8
    // AARCH64_LINUX-NEXT: str x9, [x0]
    // AARCH64_LINUX-NEXT: ldr w0, [x8]
    // AARCH64_LINUX-NEXT: ret
    // AARCH64_LINUX-NEXT: .LBB1_3
    // AARCH64_LINUX-NEXT: ldr x9, [x0, #8]
    // AARCH64_LINUX-NEXT: add x8, x9, x8
    // AARCH64_LINUX-NEXT: ldr w0, [x8]
    // AARCH64_LINUX-NEXT: ret

    // AARCH64_BE-LABEL: read_i32:
    // AARCH64_BE: ldrsw    x8, [x0, #24]
    // AARCH64_BE-NEXT: tbz w8, #31, .LBB1_2
    // AARCH64_BE-NEXT: add w9, w8, #8
    // AARCH64_BE-NEXT: cmn w8, #8
    // AARCH64_BE-NEXT: str w9, [x0, #24]
    // AARCH64_BE-NEXT: b.ls    .LBB1_3
    // AARCH64_BE-NEXT: .LBB1_2:
    // AARCH64_BE-NEXT: ldr x8, [x0]
    // AARCH64_BE-NEXT: add x9, x8, #8
    // AARCH64_BE-NEXT: str x9, [x0]
    // AARCH64_BE-NEXT: ldr w0, [x8]
    // AARCH64_BE-NEXT: ret
    // AARCH64_BE-NEXT: .LBB1_3:
    // AARCH64_BE-NEXT: ldr x9, [x0, #8]
    // AARCH64_BE-NEXT: add x8, x9, x8
    // AARCH64_BE-NEXT: ldr w0, [x8, #4]!
    // AARCH64_BE-NEXT: ret

    // ARM64EC_MSVC-LABEL: read_i32 = "#read_i32"
    // ARM64EC_MSVC: ldr x9, [x0]
    // ARM64EC_MSVC-NEXT: mov x8, x0
    // ARM64EC_MSVC-NEXT: ldr w0, [x9], #8
    // ARM64EC_MSVC-NEXT: str x9, [x8]
    // ARM64EC_MSVC-NEXT: ret

    // AARCH64_DARWIN-LABEL: _read_i32:
    // AARCH64_DARWIN: ldr x9, [x0]
    // AARCH64_DARWIN-NEXT: ldr w8, [x9], #8
    // AARCH64_DARWIN-NEXT: str x9, [x0]
    // AARCH64_DARWIN-NEXT: mov x0, x8
    // AARCH64_DARWIN-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // AARCH64_LINUX-LABEL: read_i64:
    // AARCH64_LINUX: ldrsw   x8, [x0, #24]
    // AARCH64_LINUX-NEXT: tbz w8, #31, .LBB2_2
    // AARCH64_LINUX-NEXT: add w9, w8, #8
    // AARCH64_LINUX-NEXT: cmn w8, #8
    // AARCH64_LINUX-NEXT: str w9, [x0, #24]
    // AARCH64_LINUX-NEXT: b.ls    .LBB2_3
    // AARCH64_LINUX-NEXT: .LBB2_2:
    // AARCH64_LINUX-NEXT: ldr x8, [x0]
    // AARCH64_LINUX-NEXT: add x9, x8, #8
    // AARCH64_LINUX-NEXT: str x9, [x0]
    // AARCH64_LINUX-NEXT: ldr x0, [x8]
    // AARCH64_LINUX-NEXT: ret
    // AARCH64_LINUX-NEXT: .LBB2_3
    // AARCH64_LINUX-NEXT: ldr x9, [x0, #8]
    // AARCH64_LINUX-NEXT: add x8, x9, x8
    // AARCH64_LINUX-NEXT: ldr x0, [x8]
    // AARCH64_LINUX-NEXT: ret

    // AARCH64_BE-LABEL: read_i64:
    // AARCH64_BE: ldrsw    x8, [x0, #24]
    // AARCH64_BE-NEXT: tbz w8, #31, .LBB2_2
    // AARCH64_BE-NEXT: add w9, w8, #8
    // AARCH64_BE-NEXT: cmn w8, #8
    // AARCH64_BE-NEXT: str w9, [x0, #24]
    // AARCH64_BE-NEXT: b.ls    .LBB2_3
    // AARCH64_BE-NEXT: .LBB2_2:
    // AARCH64_BE-NEXT: ldr x8, [x0]
    // AARCH64_BE-NEXT: add x9, x8, #8
    // AARCH64_BE-NEXT: str x9, [x0]
    // AARCH64_BE-NEXT: ldr x0, [x8]
    // AARCH64_BE-NEXT: ret
    // AARCH64_BE-NEXT: .LBB2_3:
    // AARCH64_BE-NEXT: ldr x9, [x0, #8]
    // AARCH64_BE-NEXT: add x8, x9, x8
    // AARCH64_BE-NEXT: ldr x0, [x8]
    // AARCH64_BE-NEXT: ret

    // ARM64EC_MSVC-LABEL: read_ptr = "#read_ptr"
    // ARM64EC_MSVC-LABEL: read_i64 = "#read_i64"
    // ARM64EC_MSVC: ldr x9, [x0]
    // ARM64EC_MSVC-NEXT: mov x8, x0
    // ARM64EC_MSVC-NEXT: ldr x0, [x9], #8
    // ARM64EC_MSVC-NEXT: str x9, [x8]
    // ARM64EC_MSVC-NEXT: ret

    // AARCH64_DARWIN-LABEL: _read_i64:
    // AARCH64_DARWIN: ldr x9, [x0]
    // AARCH64_DARWIN-NEXT: ldr x8, [x9], #8
    // AARCH64_DARWIN-NEXT: str x9, [x0]
    // AARCH64_DARWIN-NEXT: mov x0, x8
    // AARCH64_DARWIN-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i128(ap: &mut VaList<'_>) -> i128 {
    // AARCH64_LINUX-LABEL: read_i128:
    // AARCH64_LINUX: ldrsw   x8, [x0, #24]
    // AARCH64_LINUX-NEXT: tbz w8, #31, .LBB3_2
    // AARCH64_LINUX-NEXT: add x8, x8, #15
    // AARCH64_LINUX-NEXT: and x8, x8, #0xfffffffffffffff0
    // AARCH64_LINUX-NEXT: add w9, w8, #16
    // AARCH64_LINUX-NEXT: cmp w9, #0
    // AARCH64_LINUX-NEXT: str w9, [x0, #24]
    // AARCH64_LINUX-NEXT: b.le    .LBB3_3
    // AARCH64_LINUX-NEXT: .LBB3_2:
    // AARCH64_LINUX-NEXT: ldr x8, [x0]
    // AARCH64_LINUX-NEXT: add x8, x8, #15
    // AARCH64_LINUX-NEXT: and x8, x8, #0xfffffffffffffff0
    // AARCH64_LINUX-NEXT: add x9, x8, #16
    // AARCH64_LINUX-NEXT: str x9, [x0]
    // AARCH64_LINUX-NEXT: ldp x0, x1, [x8]
    // AARCH64_LINUX-NEXT: ret
    // AARCH64_LINUX-NEXT: .LBB3_3
    // AARCH64_LINUX-NEXT: ldr x9, [x0, #8]
    // AARCH64_LINUX-NEXT: add x8, x9, x8
    // AARCH64_LINUX-NEXT: ldp x0, x1, [x8]
    // AARCH64_LINUX-NEXT: ret

    // AARCH64_BE-LABEL: read_i128:
    // AARCH64_BE: ldrsw   x8, [x0, #24]
    // AARCH64_BE-NEXT: tbz     w8, #31, .LBB3_2
    // AARCH64_BE-NEXT: add     x8, x8, #15
    // AARCH64_BE-NEXT: and     x8, x8, #0xfffffffffffffff0
    // AARCH64_BE-NEXT: add     w9, w8, #16
    // AARCH64_BE-NEXT: cmp     w9, #0
    // AARCH64_BE-NEXT: str     w9, [x0, #24]
    // AARCH64_BE-NEXT: b.le    .LBB3_3
    // AARCH64_BE-NEXT: .LBB3_2:
    // AARCH64_BE-NEXT: ldr     x8, [x0]
    // AARCH64_BE-NEXT: add     x8, x8, #15
    // AARCH64_BE-NEXT: and     x8, x8, #0xfffffffffffffff0
    // AARCH64_BE-NEXT: add     x9, x8, #16
    // AARCH64_BE-NEXT: str     x9, [x0]
    // AARCH64_BE-NEXT: ldp     x0, x1, [x8]
    // AARCH64_BE-NEXT: ret
    // AARCH64_BE-NEXT: .LBB3_3:
    // AARCH64_BE-NEXT: ldr     x9, [x0, #8]
    // AARCH64_BE-NEXT: add     x8, x9, x8
    // AARCH64_BE-NEXT: ldp     x0, x1, [x8]
    // AARCH64_BE-NEXT: ret

    // ARM64EC_MSVC-LABEL: read_i128 = "#read_i128"
    // ARM64EC_MSVC: ldr x9, [x0]
    // ARM64EC_MSVC-NEXT: mov x8, x0
    // ARM64EC_MSVC-NEXT: ldp x0, x1, [x9], #16
    // ARM64EC_MSVC-NEXT: str x9, [x8]
    // ARM64EC_MSVC-NEXT: ret

    // AARCH64_DARWIN-LABEL: _read_i128:
    // AARCH64_DARWIN: ldr x8, [x0]
    // AARCH64_DARWIN-NEXT: add x8, x8, #15
    // AARCH64_DARWIN-NEXT: and x9, x8, #0xfffffffffffffff0
    // AARCH64_DARWIN-NEXT: ldr x1, [x9, #8]
    // AARCH64_DARWIN-NEXT: ldr x8, [x9], #16
    // AARCH64_DARWIN-NEXT: str x9, [x0]
    // AARCH64_DARWIN-NEXT: mov x0, x8
    // AARCH64_DARWIN-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // AARCH64_LINUX-CHECK: read_ptr = read_i64
    // AARCH64_BE-CHECK: read_ptr = read_i64
    // ARM64EC_MSVC: "#read_ptr" = "#read_i64"
    // AARCH64_DARWIN-CHECK: _read_ptr = _read_i64
    va_arg(ap)
}
