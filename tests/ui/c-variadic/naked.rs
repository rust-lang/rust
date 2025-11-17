//@ run-pass
//@ only-x86_64
//@ only-linux
#![feature(c_variadic, c_variadic_naked_functions)]

#[repr(C)]
#[derive(Debug, PartialEq)]
struct Data(i32, f64);

#[unsafe(naked)]
unsafe extern "sysv64" fn c_variadic_sysv64(_: ...) -> Data {
    // This assembly was generated with GCC, because clang/LLVM is unable to
    // optimize out the spilling of all registers to the stack.
    core::arch::naked_asm!(
        "        sub     rsp, 96",
        "        mov     QWORD PTR [rsp-88], rdi",
        "        test    al, al",
        "        je      .L7",
        "        movaps  XMMWORD PTR [rsp-40], xmm0",
        ".L7:",
        "        lea     rax, [rsp+104]",
        "        mov     rcx, QWORD PTR [rsp-40]",
        "        mov     DWORD PTR [rsp-112], 0",
        "        mov     QWORD PTR [rsp-104], rax",
        "        lea     rax, [rsp-88]",
        "        mov     QWORD PTR [rsp-96], rax",
        "        movq    xmm0, rcx",
        "        mov     eax, DWORD PTR [rsp-88]",
        "        mov     DWORD PTR [rsp-108], 48",
        "        add     rsp, 96",
        "        ret",
    )
}

#[unsafe(naked)]
unsafe extern "C" fn c_variadic_c(_: ...) -> Data {
    core::arch::naked_asm!(
        "jmp {}",
        sym c_variadic_sysv64,
    )
}

fn main() {
    unsafe {
        assert_eq!(c_variadic_sysv64(1, 2.0), Data(1, 2.0));
        assert_eq!(c_variadic_sysv64(123, 4.56), Data(123, 4.56));

        assert_eq!(c_variadic_c(1, 2.0), Data(1, 2.0));
        assert_eq!(c_variadic_c(123, 4.56), Data(123, 4.56));
    }
}
