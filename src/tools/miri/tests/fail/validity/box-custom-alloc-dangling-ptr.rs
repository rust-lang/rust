//! Ensure that a box with a custom allocator detects when the pointer is dangling.
#![feature(allocator_api)]
// This should not need the aliasing model.
//@compile-flags: -Zmiri-disable-stacked-borrows
use std::alloc::Layout;
use std::ptr::NonNull;

#[allow(unused)]
struct MyAlloc(usize, usize); // make sure `Box<T, MyAlloc>` is an `Aggregate`

unsafe impl std::alloc::Allocator for MyAlloc {
    fn allocate(&self, _layout: Layout) -> Result<NonNull<[u8]>, std::alloc::AllocError> {
        unimplemented!()
    }

    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {
        unimplemented!()
    }
}

#[repr(C)]
struct MyBox<T> {
    ptr: NonNull<T>,
    alloc: MyAlloc,
}

fn main() {
    let b = MyBox { ptr: NonNull::<i32>::dangling(), alloc: MyAlloc(0, 0) };
    let _b: Box<i32, MyAlloc> = unsafe {
        std::mem::transmute(b) //~ERROR: dangling box
    };
}
