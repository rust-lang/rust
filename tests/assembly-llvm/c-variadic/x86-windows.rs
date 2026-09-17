//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: X86_64_GNU X86_64_MSVC I686_GNU I686_MSVC
//@ [X86_64_GNU] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ [X86_64_GNU] compile-flags: --target x86_64-pc-windows-gnu
//@ [X86_64_GNU] filecheck-flags: --check-prefix X86_64
//@ [X86_64_MSVC] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ [X86_64_MSVC] compile-flags: --target x86_64-pc-windows-msvc
//@ [X86_64_MSVC] filecheck-flags: --check-prefix X86_64
//@ [I686_GNU] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ [I686_GNU] compile-flags: --target i686-pc-windows-gnu
//@ [I686_GNU] filecheck-flags: --check-prefix I686
//@ [I686_MSVC] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ [I686_MSVC] compile-flags: --target i686-pc-windows-msvc -Cforce-frame-pointers
//@ [I686_MSVC] filecheck-flags: --check-prefix I686
//@ needs-llvm-components: x86
#![feature(no_core, lang_items, intrinsics, rustc_attrs)]
#![no_core]
#![crate_type = "lib"]

// Check that the assembly that rustc generates matches what clang emits for
//
// ```c
// T function(va_list ap) {
//     return va_arg(ap, T);
// }
// ```

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
    // CHECK-LABEL: read_f64:
    //
    // X86_64: mov rax, qword ptr [rcx]
    // X86_64-NEXT: lea rdx, [rax + 8]
    // X86_64-NEXT: mov qword ptr [rcx], rdx
    // X86_64-NEXT: movsd xmm0, qword ptr [rax]
    // X86_64-NEXT: ret
    //
    // I686: mov eax, dword ptr [ebp + 8]
    // I686-NEXT: mov ecx, dword ptr [eax]
    // I686-NEXT: lea edx, [ecx + 8]
    // I686-NEXT: mov dword ptr [eax], edx
    // I686-NEXT: fld qword ptr [ecx]
    // I686-NEXT: pop ebp
    // I686-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32:
    //
    // X86_64: mov rax, qword ptr [rcx]
    // X86_64-NEXT: lea rdx, [rax + 8]
    // X86_64-NEXT: mov qword ptr [rcx], rdx
    // X86_64-NEXT: mov eax, dword ptr [rax]
    // X86_64-NEXT: ret
    //
    // I686: mov eax, dword ptr [ebp + 8]
    // I686-NEXT: mov ecx, dword ptr [eax]
    // I686-NEXT: lea edx, [ecx + 4]
    // I686-NEXT: mov dword ptr [eax], edx
    // I686-NEXT: mov eax, dword ptr [ecx]
    // I686-NEXT: pop ebp
    // I686-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64:
    //
    // X86_64: mov rax, qword ptr [rcx]
    // X86_64-NEXT: lea rdx, [rax + 8]
    // X86_64-NEXT: mov qword ptr [rcx], rdx
    // X86_64-NEXT: mov rax, qword ptr [rax]
    // X86_64-NEXT: ret
    //
    // I686: mov eax, dword ptr [ebp + 8]
    // I686-NEXT: mov ecx, dword ptr [eax]
    // I686-NEXT: lea edx, [ecx + 8]
    // I686-NEXT: mov dword ptr [eax], edx
    // I686-NEXT: mov eax, dword ptr [ecx]
    // I686-NEXT: mov edx, dword ptr [ecx + 4]
    // I686-NEXT: pop ebp
    // I686-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
#[cfg(target_pointer_width = "64")]
unsafe extern "C" fn read_i128(ap: &mut VaList<'_>) -> i128 {
    // X86_64-LABEL: read_i128:
    //
    // X86_64: mov rax, qword ptr [rcx]
    // X86_64-NEXT: lea rdx, [rax + 8]
    // X86_64-NEXT: mov qword ptr [rcx], rdx
    // X86_64-NEXT: mov rax, qword ptr [rax]
    // X86_64-NEXT: movups xmm0, xmmword ptr [rax]
    // X86_64-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // X86_64: read_ptr = read_i64
    // X86: _read_ptr = _read_i32
    va_arg(ap)
}
