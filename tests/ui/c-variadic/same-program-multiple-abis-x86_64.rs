//@ run-pass
//@ only-x86_64

// Check that multiple c-variadic calling conventions can be used in the same program.
//
// Clang and gcc reject defining functions with a non-default calling convention and a variable
// argument list, so C programs that use multiple c-variadic calling conventions are unlikely
// to come up. Here we validate that our codegen backends do in fact generate correct code.

extern "sysv64" {
    fn variadic_sysv64(_: u32, _: ...) -> u32;
}

extern "win64" {
    fn variadic_win64(_: u32, _: ...) -> u32;
}

fn main() {
    unsafe {
        assert_eq!(variadic_win64(1, 2, 3), 1 + 2 + 3);
        assert_eq!(variadic_sysv64(1, 2, 3), 1 + 2 + 3);
    }
}

// This assembly was generated using https://godbolt.org/z/dbTGanoh6, and corresponds to the
// following code compiled for the `x86_64-unknown-linux-gnu` and `x86_64-pc-windows-gnu`
// targets, respectively:
//
// ```rust
// #![feature(c_variadic)]
//
// #[unsafe(no_mangle)]
// unsafe extern "C" fn variadic(a: u32, mut args: ...) -> u32 {
//     let b = args.arg::<u32>();
//     let c = args.arg::<u32>();
//
//     a + b + c
// }
// ```
core::arch::global_asm!(
    r#"
{variadic_sysv64}:
        sub     rsp, 88
        test    al, al
        je      .LBB0_7
        movaps  xmmword ptr [rsp - 48], xmm0
        movaps  xmmword ptr [rsp - 32], xmm1
        movaps  xmmword ptr [rsp - 16], xmm2
        movaps  xmmword ptr [rsp], xmm3
        movaps  xmmword ptr [rsp + 16], xmm4
        movaps  xmmword ptr [rsp + 32], xmm5
        movaps  xmmword ptr [rsp + 48], xmm6
        movaps  xmmword ptr [rsp + 64], xmm7
.LBB0_7:
        mov     qword ptr [rsp - 88], rsi
        mov     qword ptr [rsp - 80], rdx
        mov     qword ptr [rsp - 72], rcx
        mov     qword ptr [rsp - 64], r8
        mov     qword ptr [rsp - 56], r9
        movabs  rax, 206158430216
        mov     qword ptr [rsp - 120], rax
        lea     rax, [rsp + 96]
        mov     qword ptr [rsp - 112], rax
        lea     rax, [rsp - 96]
        mov     qword ptr [rsp - 104], rax
        mov     edx, 8
        cmp     rdx, 41
        jae     .LBB0_1
        mov     rax, qword ptr [rsp - 104]
        mov     ecx, 8
        add     rcx, 8
        mov     dword ptr [rsp - 120], ecx
        mov     eax, dword ptr [rax + rdx]
        cmp     edx, 32
        ja      .LBB0_2
        add     rcx, qword ptr [rsp - 104]
        add     edx, 16
        mov     dword ptr [rsp - 120], edx
        add     eax, edi
        add     eax, dword ptr [rcx]
        add     rsp, 88
        ret
.LBB0_1:
        mov     rax, qword ptr [rsp - 112]
        lea     rcx, [rax + 8]
        mov     qword ptr [rsp - 112], rcx
        mov     eax, dword ptr [rax]
.LBB0_2:
        mov     rcx, qword ptr [rsp - 112]
        lea     rdx, [rcx + 8]
        mov     qword ptr [rsp - 112], rdx
        add     eax, edi
        add     eax, dword ptr [rcx]
        add     rsp, 88
        ret

{variadic_win64}:
        push    rax
        mov     qword ptr [rsp + 40], r9
        mov     qword ptr [rsp + 24], rdx
        mov     qword ptr [rsp + 32], r8
        lea     rax, [rsp + 40]
        mov     qword ptr [rsp], rax
        lea     eax, [rdx + rcx]
        add     eax, r8d
        pop     rcx
        ret
    "#,
    variadic_win64 = sym variadic_win64,
    variadic_sysv64 = sym variadic_sysv64,
);
