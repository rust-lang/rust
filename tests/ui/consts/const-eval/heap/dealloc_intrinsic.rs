//@ run-pass
#![feature(core_intrinsics)]
#![feature(const_heap)]

use std::intrinsics;

const _X: () = unsafe {
    let ptr = intrinsics::const_allocate(4, 4);
    intrinsics::const_deallocate(ptr, 4, 4);
};

const Y: &u32 = unsafe {
    let ptr = intrinsics::const_allocate(4, 4) as *mut u32;
    *ptr = 42;
    &*(intrinsics::const_make_global(ptr as *mut u8) as *const u32)
};

const Z: &u32 = &42;

const _Z: () = unsafe {
    let ptr1 = Y as *const _ as *mut u8;
    intrinsics::const_deallocate(ptr1, 4, 4); // nop
    intrinsics::const_deallocate(ptr1, 2, 4); // nop
    intrinsics::const_deallocate(ptr1, 4, 2); // nop

    let ptr2 = Z as *const _ as *mut u8;
    intrinsics::const_deallocate(ptr2, 4, 4); // nop
    intrinsics::const_deallocate(ptr2, 2, 4); // nop
    intrinsics::const_deallocate(ptr2, 4, 2); // nop
};

fn main() {
    assert_eq!(*Y, 42);
    assert_eq!(*Z, 42);
}
