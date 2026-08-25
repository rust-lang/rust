#![feature(allocator_api, alloc_error_hook)]

use std::alloc::*;

struct Bomb;
impl Drop for Bomb {
    fn drop(&mut self) {
        eprintln!("yes we are unwinding!");
    }
}

#[allow(unreachable_code, unused_variables)]
fn main() {
    // This is a particularly tricky hook, since it unwinds, which the default one does not.
    set_alloc_error_hook(|_layout| panic!("alloc error hook called"));

    let bomb = Bomb;
    handle_alloc_error(Layout::for_value(&0));
    std::mem::forget(bomb); // defuse unwinding bomb
}
