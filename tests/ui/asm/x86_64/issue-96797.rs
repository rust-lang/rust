// build-pass
// compile-flags: -O
// needs-asm-support
// only-x86_64
// only-linux

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
