// revisions: WIN LIN
// [WIN] only-windows
// [LIN] only-linux
// assembly-output: emit-asm
// compile-flags: --crate-type=lib -O -C llvm-args=-x86-asm-syntax=intel
// min-llvm-version: 14
// only-x86_64
// ignore-sgx
// ignore-debug

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
