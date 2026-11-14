//@ add-minicore
//@ assembly-output: emit-asm
//
//@ revisions: X86 X86_64 DARWIN WIN64 WIN32
//@ [X86] compile-flags: --target i686-unknown-linux-gnu
//@ [X86_64] compile-flags: --target x86_64-unknown-linux-gnu
//@ [DARWIN] compile-flags: --target i686-apple-darwin
//@ [WIN64] compile-flags: --target x86_64-pc-windows-gnu
//@ [WIN32] compile-flags: --target i686-pc-windows-gnu
//@ compile-flags: -Copt-level=3 --crate-type=lib
//@ needs-llvm-components: x86

#![feature(no_core)]
#![no_std]
#![no_core]

extern crate minicore;
#[cfg(target_arch = "x86")]
use minicore::arch::x86::x87_f80;
#[cfg(target_arch = "x86_64")]
use minicore::arch::x86_64::x87_f80;

// CHECK-LABEL: identity
//
// X86:      .cfi_startproc
// X86-NEXT: fldt 4(%esp)
// X86-NEXT: retl
//
// X86_64:      .cfi_startproc
// X86_64-NEXT: fldt 8(%rsp)
// X86_64-NEXT: retq
//
// DARWIN:      .cfi_startproc
// DARWIN-NEXT: pushl %ebp
// DARWIN:      movl %esp, %ebp
// DARWIN:      fldt 8(%ebp)
// DARWIN-NEXT: popl %ebp
// DARWIN-NEXT: retl
//
// WIN64:      movq %rcx, %rax
// WIN64-NEXT: movaps (%rdx), %xmm0
// WIN64-NEXT: movaps %xmm0, (%rcx)
// WIN64-NEXT: retq
//
// WIN32:      .cfi_startproc
// WIN32-NEXT: pushl %ebp
// WIN32:      movl %esp, %ebp
// WIN32:      fldt 8(%ebp)
// WIN32-NEXT: popl %ebp
// WIN32-NEXT: retl
#[unsafe(no_mangle)]
pub extern "C" fn identity(x: x87_f80) -> x87_f80 {
    x
}

// CHECK-LABEL: mixed
//
// X86:      .cfi_startproc
// X86-NEXT: movl 24(%esp), %eax
// X86-NEXT: movsd 4(%esp), %xmm0
// X86-NEXT: movsd %xmm0, (%eax)
// X86-NEXT: retl
//
// X86_64:      .cfi_startproc
// X86_64-NEXT: movsd 8(%rsp), %xmm0
// X86_64-NEXT: movsd %xmm0, (%rdi)
// X86_64-NEXT: retq
//
// DARWIN:      .cfi_startproc
// DARWIN-NEXT: pushl %ebp
// DARWIN:      movl %esp, %ebp
// DARWIN:      movl 40(%ebp), %eax
// DARWIN-NEXT: movsd 8(%ebp), %xmm0
// DARWIN-NEXT: movsd %xmm0, (%eax)
// DARWIN-NEXT: popl %ebp
// DARWIN-NEXT: retl
//
// WIN64:      movsd (%rcx), %xmm0
// WIN64-NEXT: movsd %xmm0, (%rdx)
// WIN64-NEXT: retq
//
// WIN32:      .cfi_startproc
// WIN32-NEXT: pushl %ebp
// WIN32:      movl %esp, %ebp
// WIN32:      movl 32(%ebp), %eax
// WIN32-NEXT: movsd 8(%ebp), %xmm0
// WIN32-NEXT: movsd %xmm0, (%eax)
// WIN32-NEXT: popl %ebp
// WIN32-NEXT: retl
#[unsafe(no_mangle)]
pub extern "C" fn mixed(value: Mixed, res: &mut f64) {
    // Write the value to a pointer instead of returning it to avoid the extra `f32`/`f64` return
    // assembly on 32-bit.
    *res = value.a;
}

#[repr(C)]
struct Mixed {
    a: f64,
    b: x87_f80,
}

// CHECK-LABEL: pair
//
// X86:      .cfi_startproc
// X86-NEXT: fldt 16(%esp)
// X86-NEXT: retl
//
// X86_64:      .cfi_startproc
// X86_64-NEXT: fldt 24(%rsp)
// X86_64-NEXT: retq
//
// DARWIN:      .cfi_startproc
// DARWIN-NEXT: pushl %ebp
// DARWIN:      movl %esp, %ebp
// DARWIN:      fldt 24(%ebp)
// DARWIN-NEXT: popl %ebp
// DARWIN-NEXT: retl
//
// WIN64:      movq %rcx, %rax
// WIN64-NEXT: movaps 16(%rdx), %xmm0
// WIN64-NEXT: movaps %xmm0, (%rcx)
// WIN64-NEXT: retq
//
// WIN32:      .cfi_startproc
// WIN32-NEXT: pushl %ebp
// WIN32:      movl %esp, %ebp
// WIN32:      fldt 20(%ebp)
// WIN32-NEXT: popl %ebp
// WIN32-NEXT: retl
#[unsafe(no_mangle)]
pub extern "C" fn pair(value: Pair) -> x87_f80 {
    value.b
}

#[repr(C)]
struct Pair {
    a: x87_f80,
    b: x87_f80,
}

// CHECK-LABEL: quad
//
// X86:      .cfi_startproc
// X86-NEXT: fldt 40(%esp)
// X86-NEXT: retl
//
// X86_64:      .cfi_startproc
// X86_64-NEXT: fldt 56(%rsp)
// X86_64-NEXT: retq
//
// DARWIN:      .cfi_startproc
// DARWIN-NEXT: pushl %ebp
// DARWIN:      movl %esp, %ebp
// DARWIN:      fldt 56(%ebp)
// DARWIN-NEXT: popl %ebp
// DARWIN-NEXT: retl
//
// WIN64:      movq %rcx, %rax
// WIN64-NEXT: movaps 48(%rdx), %xmm0
// WIN64-NEXT: movaps %xmm0, (%rcx)
// WIN64-NEXT: retq
//
// WIN32:      .cfi_startproc
// WIN32-NEXT: pushl %ebp
// WIN32:      movl %esp, %ebp
// WIN32:      fldt 44(%ebp)
// WIN32-NEXT: popl %ebp
// WIN32-NEXT: retl
#[unsafe(no_mangle)]
pub extern "C" fn quad(value: Quad) -> x87_f80 {
    value.d
}

#[repr(C)]
struct Quad {
    a: x87_f80,
    b: x87_f80,
    c: x87_f80,
    d: x87_f80,
}
