// build-pass
//@compile-flags: -O
//@needs-asm-support
//@only-target-x86_64
//@only-target-linux

// regression test for #96797

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
