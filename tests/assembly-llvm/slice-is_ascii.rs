//@ revisions: WIN LIN
//@ [WIN] only-windows
//@ [LIN] only-linux
//@ assembly-output: emit-asm
//@ compile-flags: --crate-type=lib -Copt-level=3 -C llvm-args=-x86-asm-syntax=intel
//@ only-x86_64
//@ ignore-sgx

// No longer optimizes as desired, tracked at:
// https://github.com/rust-lang/rust/issues/154141
// https://github.com/llvm/llvm-project/issues/209216
//@ max-llvm-major-version: 22

#![feature(str_internals)]

// CHECK-LABEL: is_ascii_simple_demo:
#[no_mangle]
pub fn is_ascii_simple_demo(bytes: &[u8]) -> bool {
    // Linux (System V): pointer is rdi; length is rsi
    // Windows: pointer is rcx; length is rdx.

    // CHECK-NOT: mov
    // CHECK-NOT: test
    // CHECK-NOT: cmp

    // CHECK: .[[LOOPHEAD:.+]]:
    // CHECK-NEXT: mov [[TEMP:.+]], [[LEN:rsi|rdx]]
    // CHECK-NEXT: sub [[LEN]], 1
    // CHECK-NEXT: jb .[[LOOPEXIT:.+]]
    // CHECK-NEXT: cmp byte ptr [{{rdi|rcx}} + [[TEMP]] - 1], 0
    // CHECK-NEXT: jns .[[LOOPHEAD]]

    // CHECK-NEXT: .[[LOOPEXIT]]:
    // CHECK-NEXT: test [[TEMP]], [[TEMP]]
    // CHECK-NEXT: sete al
    // CHECK-NEXT: ret
    core::slice::is_ascii_simple(bytes)
}
