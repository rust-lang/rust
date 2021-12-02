// only-x86_64
// assembly-output: emit-asm
// compile-flags: -C llvm-args=--x86-asm-syntax=intel

#![feature(global_asm, asm_const)]
#![crate_type = "rlib"]

// CHECK: mov eax, eax
global_asm!("mov eax, eax");
// CHECK: mov ebx, 5
global_asm!("mov ebx, {}", const 5);
// CHECK: mov ecx, 5
global_asm!("movl ${}, %ecx", const 5, options(att_syntax));
