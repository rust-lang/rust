// Test that `extern "custom"` functions can be called from assembly, and defined using a naked
// function, and `global_asm!` with an `extern "custom"` block.
//
//@ run-pass
//@ only-x86_64
#![feature(abi_custom)]

use std::arch::{asm, global_asm, naked_asm};

#[unsafe(naked)]
unsafe extern "custom" fn double(a: u64) -> u64 {
    naked_asm!("add rax, rax", "ret");
}

global_asm!(
    "    .globl  increment",
    "    .type   increment,@function",
    "increment:",
    "    .cfi_startproc",
    "    add     rax, 1",
    "    ret",
    ".Lfunc_end0:",
    "    .size   increment, .Lfunc_end0-increment",
    "    .cfi_endproc",
);

unsafe extern "custom" {
    fn increment(a: u64) -> u64;
}

#[repr(transparent)]
struct Thing(u64);

impl Thing {
    #[unsafe(naked)]
    unsafe extern "custom" fn is_even(self) -> bool {
        naked_asm!("test al, 1", "sete al", "ret");
    }
}

trait BitwiseNot {
    #[unsafe(naked)]
    unsafe extern "custom" fn bitwise_not(a: u64) -> u64 {
        naked_asm!("not rax", "ret");
    }
}

impl BitwiseNot for Thing {}

pub fn main() {
    let mut x: u64 = 21;
    unsafe { asm!("call {}", sym double, inout("rax") x) };
    assert_eq!(x, 42);

    let mut x: u64 = 41;
    unsafe { asm!("call {}", sym increment, inout("rax") x) };
    assert_eq!(x, 42);

    let mut x: u32;
    unsafe { asm!("call {}", sym Thing::is_even, inout("rax") 42 => x) };
    assert!(x != 0);

    let mut x: u64 = 42;
    unsafe { asm!("call {}", sym Thing::bitwise_not, inout("rax") x) };
    assert_eq!(x, !42);

    // Create and call in `asm!` an `extern "custom"` function pointer.
    fn caller(f: unsafe extern "custom" fn(u64) -> u64, mut x: u64) -> u64 {
        unsafe { asm!("call {}", in(reg) f, inout("rax") x) };
        x
    }

    assert_eq!(caller(double, 2), 4);
}
