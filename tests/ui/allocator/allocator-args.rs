use std::alloc::{GlobalAlloc, Layout};

struct A;

unsafe impl GlobalAlloc for A {
    unsafe fn alloc(&self, _: Layout) -> *mut u8 { panic!() }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) { panic!() }
}

#[global_allocator(malloc)] //~ ERROR malformed `global_allocator` attribute input
static S: A = A;

fn main() {}
