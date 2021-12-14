// only-x86_64
// assembly-output: emit-asm
// compile-flags: -C llvm-args=--x86-asm-syntax=intel

#![feature(asm_const)]
#![crate_type = "rlib"]

use std::arch::global_asm;

// CHECK: mov eax, eax
global_asm!("mov eax, eax");
// CHECK: mov ebx, 5
global_asm!("mov ebx, {}", const 5);
// CHECK: mov ecx, 5
global_asm!("movl ${}, %ecx", const 5, options(att_syntax));
