use std::alloc::{GlobalAlloc, Layout};

struct Test(u32);

unsafe impl GlobalAlloc for Test {
    unsafe fn alloc(&self, _layout: Layout) -> *mut u8 {
        self.0 += 1; //~ ERROR cannot assign
        0 as *mut u8
    }

    unsafe fn dealloc(&self, _ptr: *mut u8, _layout: Layout) {
        unimplemented!();
    }
}

fn main() { }
