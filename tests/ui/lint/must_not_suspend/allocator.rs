//@ edition: 2021

#![feature(must_not_suspend, allocator_api)]
#![deny(must_not_suspend)]

use std::alloc::*;
use std::ptr::NonNull;

#[must_not_suspend]
struct MyAllocatorWhichMustNotSuspend;

unsafe impl Allocator for MyAllocatorWhichMustNotSuspend {
    fn allocate(&self, l: Layout) -> Result<NonNull<[u8]>, AllocError> {
        Global.allocate(l)
    }
    unsafe fn deallocate(&self, p: NonNull<u8>, l: Layout) {
        Global.deallocate(p, l)
    }
}

async fn suspend() {}

async fn foo() {
    let x = Box::new_in(1i32, MyAllocatorWhichMustNotSuspend);
    //~^ ERROR allocator `MyAllocatorWhichMustNotSuspend` held across a suspend point, but should not be
    suspend().await;
    drop(x);
}

fn main() {}
