//@run-pass
#![feature(sync_unsafe_cell)]

use std::cell::SyncUnsafeCell;
use std::ptr;

#[repr(C)]
struct SyncPtr {
    foo: *mut u32,
}
unsafe impl Sync for SyncPtr {}

static mut STATIC: u32 = 42;

static INTERIOR_MUTABLE_STATIC: SyncUnsafeCell<u32> = SyncUnsafeCell::new(42);

// A static that mutably points to STATIC.
static PTR: SyncPtr = SyncPtr { foo: ptr::addr_of_mut!(STATIC) };
static INTERIOR_MUTABLE_PTR: SyncPtr =
    SyncPtr { foo: ptr::addr_of!(INTERIOR_MUTABLE_STATIC) as *mut u32 };

fn main() {
    let ptr = PTR.foo;
    unsafe {
        assert_eq!(*ptr, 42);
        *ptr = 0;
        assert_eq!(*PTR.foo, 0);
    }

    let ptr = INTERIOR_MUTABLE_PTR.foo;
    unsafe {
        assert_eq!(*ptr, 42);
        *ptr = 0;
        assert_eq!(*INTERIOR_MUTABLE_PTR.foo, 0);
    }
}
