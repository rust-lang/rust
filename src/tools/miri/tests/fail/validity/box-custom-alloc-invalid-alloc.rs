//! Ensure that a box with a custom allocator detects when the allocator itself is invalid.
#![feature(allocator_api)]
// This should not need the aliasing model.
//@compile-flags: -Zmiri-disable-stacked-borrows
use std::alloc::Layout;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

// make sure `Box<T, MyAlloc>` is an `Aggregate`
#[allow(unused)]
struct MyAlloc {
    my_alloc_field1: usize,
    my_alloc_field2: usize,
}

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
    alloc: MaybeUninit<MyAlloc>,
}

fn main() {
    let b = MyBox { ptr: NonNull::from(&42), alloc: MaybeUninit::uninit() };
    let _b: Box<i32, MyAlloc> = unsafe {
        std::mem::transmute(b) //~ERROR: uninitialized memory
    };
}
