// build-pass
// compile-flags: -O
// min-llvm-version: 14.0.5

// regression test for #96797

#![feature(asm_sym)]

use std::arch::global_asm;

#[no_mangle]
fn my_func() {}

global_asm!("call_foobar: jmp {}", sym foobar);

fn foobar() {}

fn main() {
    extern "Rust" {
        fn call_foobar();
    }
    unsafe { call_foobar() };
}
