//@compile-flags: -Zoom=panic
#![feature(allocator_api)]

use std::alloc::*;

struct Bomb;
impl Drop for Bomb {
    fn drop(&mut self) {
        eprintln!("yes we are unwinding!");
    }
}

#[allow(unreachable_code, unused_variables)]
fn main() {
    let bomb = Bomb;
    handle_alloc_error(Layout::for_value(&0));
    std::mem::forget(bomb); // defuse unwinding bomb
}
