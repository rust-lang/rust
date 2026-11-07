// Test that we're properly monomorphizing sym args in global asm blocks
// that point to associated items.

//@ edition: 2021
//@ needs-asm-support
//@ only-x86_64-unknown-linux-gnu
//@ build-pass

#![no_main]

use std::arch::global_asm;

fn foo() {
    loop {}
}

trait Foo {
    fn bar();
}

impl Foo for i32 {
    fn bar() {
        loop {}
    }
}

global_asm!(".global main", "main:", "call {}", sym <i32 as Foo>::bar);
