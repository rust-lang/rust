//@ run-fail
//@ ignore-wasm32-bare: No panic messages
//@ compile-flags: -C debug-assertions -Zmir-opt-level=0
//@ error-pattern: occupied niche: found 0x0 but must be in 0x1..=0xffffffff

use std::ptr::NonNull;

fn main() {
    unsafe {
        std::mem::transmute::<(usize, *const u8), (usize, NonNull<u8>)>((0usize, std::ptr::null()));
    }
}
