//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: WINDOWS_GNU WINDOWS_MSVC
//@ [WINDOWS_GNU] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ [WINDOWS_GNU] compile-flags: --target x86_64-pc-windows-gnu
//@ [WINDOWS_GNU] needs-llvm-components: x86
//@ [WINDOWS_MSVC] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ [WINDOWS_MSVC] compile-flags: --target x86_64-pc-windows-msvc
//@ [WINDOWS_MSVC] needs-llvm-components: x86
#![feature(c_variadic, no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what clang emits.

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
    // CHECK-LABEL: read_f64:
    // CHECK: mov rax, qword ptr [rcx]
    // CHECK-NEXT: lea rdx, [rax + 8]
    // CHECK-NEXT: mov qword ptr [rcx], rdx
    // CHECK-NEXT: movsd xmm0, qword ptr [rax]
    // CHECK-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32:
    // CHECK: mov rax, qword ptr [rcx]
    // CHECK-NEXT: lea rdx, [rax + 8]
    // CHECK-NEXT: mov qword ptr [rcx], rdx
    // CHECK-NEXT: mov eax, dword ptr [rax]
    // CHECK-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64:
    // CHECK: mov rax, qword ptr [rcx]
    // CHECK-NEXT: lea rdx, [rax + 8]
    // CHECK-NEXT: mov qword ptr [rcx], rdx
    // CHECK-NEXT: mov rax, qword ptr [rax]
    // CHECK-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i128(ap: &mut VaList<'_>) -> i128 {
    // CHECK-LABEL: read_i128:
    // CHECK: mov rax, qword ptr [rcx]
    // CHECK-NEXT: lea rdx, [rax + 8]
    // CHECK-NEXT: mov qword ptr [rcx], rdx
    // CHECK-NEXT: mov rax, qword ptr [rax]
    // CHECK-NEXT: movups xmm0, xmmword ptr [rax]
    // CHECK-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // CHECK: read_ptr = read_i64
    va_arg(ap)
}
