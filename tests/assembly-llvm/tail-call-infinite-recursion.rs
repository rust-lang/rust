//@ add-minicore
//@ assembly-output: emit-asm
//@ compile-flags: --target x86_64-unknown-linux-gnu -Copt-level=0 -Cllvm-args=-x86-asm-syntax=intel
//@ needs-llvm-components: x86
#![expect(incomplete_features)]
#![feature(no_core, explicit_tail_calls)]
#![crate_type = "lib"]
#![no_core]

extern crate minicore;
use minicore::*;

// Test that an infinite loop via guaranteed tail calls does not blow the stack.

// CHECK-LABEL: inf
// CHECK: mov rax, qword ptr [rip + inf@GOTPCREL]
// CHECK: jmp rax
#[unsafe(no_mangle)]
fn inf() -> ! {
    become inf()
}
