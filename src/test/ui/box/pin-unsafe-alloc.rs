#![feature(allocator_api)]
#![feature(box_into_pin)]

use std::alloc::{AllocError, Allocator, Layout, System};
use std::ptr::NonNull;
use std::marker::PhantomPinned;
use std::boxed::Box;

struct Alloc {}

unsafe impl Allocator for Alloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        System.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        System.deallocate(ptr, layout)
    }
}

fn main() {
    struct MyPinned {
        _value: u32,
        _pinned: PhantomPinned,
    }

    let value = MyPinned {
        _value: 0,
        _pinned: PhantomPinned,
    };
    let alloc = Alloc {};
    let _ = Box::pin_in(value, alloc);
    //~^ ERROR the trait bound `Alloc: PinSafeAllocator` is not satisfied
}
