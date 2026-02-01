//@ add-minicore
//@ assembly-output: emit-asm
//@ needs-llvm-components: x86
//@ compile-flags: --target=x86_64-unknown-linux-gnu
//@ compile-flags: -Copt-level=3 -C llvm-args=-x86-asm-syntax=intel

#![feature(no_core, explicit_tail_calls)]
#![expect(incomplete_features)]
#![no_core]
#![crate_type = "lib"]

// Test tail calls with `PassMode::Indirect { on_stack: false, .. }` arguments.
//
// Normally an indirect argument with `on_stack: false` would be passed as a pointer to the
// caller's stack frame. For tail calls, that would be unsound, because the caller's stack
// frame is overwritten by the callee's stack frame.
//
// The solution is to write the argument into the caller's argument place (stored somewhere further
// up the stack), and forward that place.

extern crate minicore;
use minicore::*;

#[repr(C)]
struct S {
    x: u64,
    y: u64,
    z: u64,
}

unsafe extern "C" {
    safe fn force_usage(_: u64, _: u64, _: u64) -> u64;
}

// CHECK-LABEL: callee:
// CHECK-NEXT: .cfi_startproc
//
// CHECK-NEXT: mov rax, qword ptr [rdi]
// CHECK-NEXT: mov rsi, qword ptr [rdi + 8]
// CHECK-NEXT: mov rdx, qword ptr [rdi + 16]
// CHECK-NEXT: mov rdi, rax
//
// CHECK-NEXT: jmp qword ptr [rip + force_usage@GOTPCREL]
#[inline(never)]
#[unsafe(no_mangle)]
fn callee(s: S) -> u64 {
    force_usage(s.x, s.y, s.z)
}

// CHECK-LABEL: caller1:
// CHECK-NEXT: .cfi_startproc
//
// Just forward the argument:
//
// CHECK-NEXT: jmp qword ptr [rip + callee@GOTPCREL]
#[unsafe(no_mangle)]
fn caller1(s: S) -> u64 {
    become callee(s);
}

// CHECK-LABEL: caller2:
// CHECK-NEXT: .cfi_startproc
//
// Construct the S value directly into the argument slot:
//
// CHECK-NEXT: mov qword ptr [rdi], 1
// CHECK-NEXT: mov qword ptr [rdi + 8], 2
// CHECK-NEXT: mov qword ptr [rdi + 16], 3
//
// CHECK-NEXT: jmp qword ptr [rip + callee@GOTPCREL]
#[unsafe(no_mangle)]
fn caller2(_: S) -> u64 {
    let s = S { x: 1, y: 2, z: 3 };
    become callee(s);
}
