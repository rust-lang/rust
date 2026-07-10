//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: X86_64 X86_64_GNUX32 I686
//@ [X86_64] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ [X86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [X86_64] needs-llvm-components: x86
//@ [X86_64_GNUX32] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ [X86_64_GNUX32] compile-flags: --target x86_64-unknown-linux-gnux32
//@ [X86_64_GNUX32] needs-llvm-components: x86
//@ [I686] compile-flags: -Copt-level=3 -Cllvm-args=-x86-asm-syntax=intel
//@ [I686] compile-flags: --target i686-unknown-linux-gnu
//@ [I686] needs-llvm-components: x86
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

    // X86_64: mov     ecx, dword ptr [rdi + 4]
    // X86_64-NEXT: cmp     rcx, 160
    // X86_64-NEXT: ja      .LBB0_2
    // X86_64-NEXT: mov     rax, rcx
    // X86_64-NEXT: add     rax, qword ptr [rdi + 16]
    // X86_64-NEXT: add     ecx, 16
    // X86_64-NEXT: mov     dword ptr [rdi + 4], ecx
    // X86_64-NEXT: movsd   xmm0, qword ptr [rax]
    // X86_64-NEXT: ret
    // X86_64-NEXT: .LBB0_2:
    // X86_64-NEXT: mov     rax, qword ptr [rdi + 8]
    // X86_64-NEXT: lea     rcx, [rax + 8]
    // X86_64-NEXT: mov     qword ptr [rdi + 8], rcx
    // X86_64-NEXT: movsd   xmm0, qword ptr [rax]
    // X86_64-NEXT: ret

    // X86_64-NEXT_GNUX32: mov     ecx, dword ptr [edi + 4]
    // X86_64-NEXT_GNUX32-NEXT: cmp     ecx, 160
    // X86_64-NEXT_GNUX32-NEXT: ja      .LBB0_2
    // X86_64-NEXT_GNUX32-NEXT: mov     eax, dword ptr [edi + 12]
    // X86_64-NEXT_GNUX32-NEXT: add     eax, ecx
    // X86_64-NEXT_GNUX32-NEXT: add     ecx, 16
    // X86_64-NEXT_GNUX32-NEXT: mov     dword ptr [edi + 4], ecx
    // X86_64-NEXT_GNUX32-NEXT: movsd   xmm0, qword ptr [eax]
    // X86_64-NEXT_GNUX32-NEXT: ret
    // X86_64-NEXT_GNUX32-NEXT: .LBB0_2:
    // X86_64-NEXT_GNUX32-NEXT: mov     eax, dword ptr [edi + 8]
    // X86_64-NEXT_GNUX32-NEXT: lea     ecx, [rax + 8]
    // X86_64-NEXT_GNUX32-NEXT: mov     dword ptr [edi + 8], ecx
    // X86_64-NEXT_GNUX32-NEXT: movsd   xmm0, qword ptr [eax]
    // X86_64-NEXT_GNUX32-NEXT: ret

    // I686: mov eax, dword ptr [esp + 4]
    // I686-NEXT: mov ecx, dword ptr [eax]
    // I686-NEXT: lea edx, [ecx + 8]
    // I686-NEXT: mov dword ptr [eax], edx
    // I686-NEXT: fld qword ptr [ecx]
    // I686-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i32(ap: &mut VaList<'_>) -> i32 {
    // CHECK-LABEL: read_i32
    //
    // X86_64: mov     ecx, dword ptr [rdi]
    // X86_64-NEXT: cmp     rcx, 40
    // X86_64-NEXT: ja      .LBB1_2
    // X86_64-NEXT: mov     rax, rcx
    // X86_64-NEXT: add     rax, qword ptr [rdi + 16]
    // X86_64-NEXT: add     ecx, 8
    // X86_64-NEXT: mov     dword ptr [rdi], ecx
    // X86_64-NEXT: mov     eax, dword ptr [rax]
    // X86_64-NEXT: ret
    // X86_64-NEXT: .LBB1_2:
    // X86_64-NEXT: mov     rax, qword ptr [rdi + 8]
    // X86_64-NEXT: lea     rcx, [rax + 8]
    // X86_64-NEXT: mov     qword ptr [rdi + 8], rcx
    // X86_64-NEXT: mov     eax, dword ptr [rax]
    // X86_64-NEXT: ret

    // X86_64-NEXT_GNUX32: mov     ecx, dword ptr [edi]
    // X86_64-NEXT_GNUX32-NEXT: cmp     ecx, 40
    // X86_64-NEXT_GNUX32-NEXT: ja      .LBB1_2
    // X86_64-NEXT_GNUX32-NEXT: mov     eax, dword ptr [edi + 12]
    // X86_64-NEXT_GNUX32-NEXT: add     eax, ecx
    // X86_64-NEXT_GNUX32-NEXT: add     ecx, 8
    // X86_64-NEXT_GNUX32-NEXT: mov     dword ptr [edi], ecx
    // X86_64-NEXT_GNUX32-NEXT: mov     eax, dword ptr [eax]
    // X86_64-NEXT_GNUX32-NEXT: ret
    // X86_64-NEXT_GNUX32-NEXT: .LBB1_2:
    // X86_64-NEXT_GNUX32-NEXT: mov     eax, dword ptr [edi + 8]
    // X86_64-NEXT_GNUX32-NEXT: lea     ecx, [rax + 8]
    // X86_64-NEXT_GNUX32-NEXT: mov     dword ptr [edi + 8], ecx
    // X86_64-NEXT_GNUX32-NEXT: mov     eax, dword ptr [eax]
    // X86_64-NEXT_GNUX32-NEXT: ret

    // I686: mov eax, dword ptr [esp + 4]
    // I686-NEXT: mov ecx, dword ptr [eax]
    // I686-NEXT: lea edx, [ecx + 4]
    // I686-NEXT: mov dword ptr [eax], edx
    // I686-NEXT: mov eax, dword ptr [ecx]
    // I686-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_i64(ap: &mut VaList<'_>) -> i64 {
    // CHECK-LABEL: read_i64

    // X86_64: mov     ecx, dword ptr [rdi]
    // X86_64-NEXT: cmp     rcx, 40
    // X86_64-NEXT: ja      .LBB2_2
    // X86_64-NEXT: mov     rax, rcx
    // X86_64-NEXT: add     rax, qword ptr [rdi + 16]
    // X86_64-NEXT: add     ecx, 8
    // X86_64-NEXT: mov     dword ptr [rdi], ecx
    // X86_64-NEXT: mov     rax, qword ptr [rax]
    // X86_64-NEXT: ret
    // X86_64-NEXT: .LBB2_2:
    // X86_64-NEXT: mov     rax, qword ptr [rdi + 8]
    // X86_64-NEXT: lea     rcx, [rax + 8]
    // X86_64-NEXT: mov     qword ptr [rdi + 8], rcx
    // X86_64-NEXT: mov     rax, qword ptr [rax]
    // X86_64-NEXT: ret

    // X86_64-NEXT_GNUX32: mov     ecx, dword ptr [edi]
    // X86_64-NEXT_GNUX32-NEXT: cmp     ecx, 40
    // X86_64-NEXT_GNUX32-NEXT: ja      .LBB2_2
    // X86_64-NEXT_GNUX32-NEXT: mov     eax, dword ptr [edi + 12]
    // X86_64-NEXT_GNUX32-NEXT: add     eax, ecx
    // X86_64-NEXT_GNUX32-NEXT: add     ecx, 8
    // X86_64-NEXT_GNUX32-NEXT: mov     dword ptr [edi], ecx
    // X86_64-NEXT_GNUX32-NEXT: mov     rax, qword ptr [eax]
    // X86_64-NEXT_GNUX32-NEXT: ret
    // X86_64-NEXT_GNUX32-NEXT: .LBB2_2:
    // X86_64-NEXT_GNUX32-NEXT: mov     eax, dword ptr [edi + 8]
    // X86_64-NEXT_GNUX32-NEXT: lea     ecx, [rax + 8]
    // X86_64-NEXT_GNUX32-NEXT: mov     dword ptr [edi + 8], ecx
    // X86_64-NEXT_GNUX32-NEXT: mov     rax, qword ptr [eax]
    // X86_64-NEXT_GNUX32-NEXT: ret

    // I686: mov eax, dword ptr [esp + 4]
    // I686-NEXT: mov ecx, dword ptr [eax]
    // I686-NEXT: lea edx, [ecx + 8]
    // I686-NEXT: mov dword ptr [eax], edx
    // I686-NEXT: mov eax, dword ptr [ecx]
    // I686-NEXT: mov edx, dword ptr [ecx + 4]
    // I686-NEXT: ret
    va_arg(ap)
}

#[unsafe(no_mangle)]
#[cfg(target_pointer_width = "64")]
unsafe extern "C" fn read_i128(ap: &mut VaList<'_>) -> i128 {
    // X86_64-LABEL: read_i128
    //
    // X86_64: mov ecx, dword ptr [rdi]
    // X86_64-NEXT: cmp rcx, 32
    // X86_64-NEXT: ja  .LBB3_2
    // X86_64-NEXT: mov rdx, qword ptr [rdi + 16]
    // X86_64-NEXT: mov rax, qword ptr [rdx + rcx]
    // X86_64-NEXT: mov rdx, qword ptr [rdx + rcx + 8]
    // X86_64-NEXT: add ecx, 16
    // X86_64-NEXT: mov dword ptr [rdi], ecx
    // X86_64-NEXT: ret
    // X86_64-NEXT: .LBB3_2:
    // X86_64-NEXT: mov     rcx, qword ptr [rdi + 8]
    // X86_64-NEXT: add     rcx, 15
    // X86_64-NEXT: and     rcx, -16
    // X86_64-NEXT: lea     rax, [rcx + 16]
    // X86_64-NEXT: mov     qword ptr [rdi + 8], rax
    // X86_64-NEXT: mov     rax, qword ptr [rcx]
    // X86_64-NEXT: mov     rdx, qword ptr [rcx + 8]
    // X86_64-NEXT: ret

    // X86_64-NEXT_GNUX32: mov     ecx, dword ptr [edi]
    // X86_64-NEXT_GNUX32-NEXT: cmp     ecx, 32
    // X86_64-NEXT_GNUX32-NEXT: ja      .LBB3_2
    // X86_64-NEXT_GNUX32-NEXT: mov     edx, dword ptr [edi + 12]
    // X86_64-NEXT_GNUX32-NEXT: mov     rax, qword ptr [edx + ecx]
    // X86_64-NEXT_GNUX32-NEXT: mov     rdx, qword ptr [edx + ecx + 8]
    // X86_64-NEXT_GNUX32-NEXT: add     ecx, 16
    // X86_64-NEXT_GNUX32-NEXT: mov     dword ptr [edi], ecx
    // X86_64-NEXT_GNUX32-NEXT: ret
    // X86_64-NEXT_GNUX32-NEXT: .LBB3_2:
    // X86_64-NEXT_GNUX32-NEXT: mov     ecx, dword ptr [edi + 8]
    // X86_64-NEXT_GNUX32-NEXT: add     ecx, 15
    // X86_64-NEXT_GNUX32-NEXT: and     ecx, -16
    // X86_64-NEXT_GNUX32-NEXT: lea     eax, [rcx + 16]
    // X86_64-NEXT_GNUX32-NEXT: mov     dword ptr [edi + 8], eax
    // X86_64-NEXT_GNUX32-NEXT: mov     rax, qword ptr [ecx]
    // X86_64-NEXT_GNUX32-NEXT: mov     rdx, qword ptr [ecx + 8]
    // X86_64-NEXT_GNUX32-NEXT: ret

    va_arg(ap)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn read_ptr(ap: &mut VaList<'_>) -> *const u8 {
    // X86_64: read_ptr = read_i64
    // X86_64_GNUX32: read_ptr = read_i32
    // I686: read_ptr = read_i32
    va_arg(ap)
}
