//@ add-minicore
//@ build-pass
//@ compile-flags: --target x86_64-unknown-linux-gnu -O
//@ needs-llvm-components: x86
//@ ignore-backends: gcc
#![crate_type = "lib"]
#![feature(no_core)]
#![no_core]

extern crate minicore;
use minicore::*;

// regression test for #96797

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
