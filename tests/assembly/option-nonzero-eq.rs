// revisions: WIN LIN
// [WIN] only-windows
// [LIN] only-linux
// assembly-output: emit-asm
// compile-flags: --crate-type=lib -O -C llvm-args=-x86-asm-syntax=intel
// only-x86_64
// ignore-sgx
// ignore-debug

use std::cmp::Ordering;

// CHECK-lABEL: ordering_eq:
#[no_mangle]
pub fn ordering_eq(l: Option<Ordering>, r: Option<Ordering>) -> bool {
    // Linux (System V): first two arguments are rdi then rsi
    // Windows: first two arguments are rcx then rdx
    // Both use rax for the return value.

    // CHECK-NOT: mov
    // CHECK-NOT: test
    // CHECK-NOT: cmp

    // LIN: cmp dil, sil
    // WIN: cmp cl, dl
    // CHECK-NEXT: sete al
    // CHECK-NEXT: ret
    l == r
}
