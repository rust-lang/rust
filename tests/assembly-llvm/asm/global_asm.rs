//@ only-x86_64
//@ only-linux
//@ assembly-output: emit-asm
//@ compile-flags: -C llvm-args=--x86-asm-syntax=intel
//@ compile-flags: -C symbol-mangling-version=v0

#![crate_type = "rlib"]

use std::arch::global_asm;

#[no_mangle]
fn my_func() {}

#[no_mangle]
static MY_STATIC: i32 = 0;

// CHECK: mov eax, eax
global_asm!("mov eax, eax");
// CHECK: mov ebx, 5
global_asm!("mov ebx, {}", const 5);
// CHECK: mov ecx, 5
global_asm!("movl ${}, %ecx", const 5, options(att_syntax));
// CHECK: call my_func
global_asm!("call {}", sym my_func);
// CHECK: lea rax, [rip + MY_STATIC]
global_asm!("lea rax, [rip + {}]", sym MY_STATIC);
// CHECK: call _RNvC[[CRATE_IDENT:[a-zA-Z0-9]{12}]]_10global_asm6foobar
global_asm!("call {}", sym foobar);
// CHECK: _RNvC[[CRATE_IDENT]]_10global_asm6foobar:
fn foobar() {
    loop {}
}
