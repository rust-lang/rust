//@ build-pass
//@ compile-flags: -O
//@ needs-asm-support
//@ only-x86_64
//@ only-linux

// regression test for #96797

use std::arch::global_asm;

#[no_mangle]
fn my_func() {}

global_asm!("
.globl call_foobar
.type call_foobar,@function
.pushsection .text.call_foobar,\"ax\",@progbits
call_foobar: jmp {}
.size call_foobar, .-call_foobar
.popsection
", sym foobar);

fn foobar() {}

fn main() {
    extern "Rust" {
        fn call_foobar();
    }
    unsafe { call_foobar() };
}
