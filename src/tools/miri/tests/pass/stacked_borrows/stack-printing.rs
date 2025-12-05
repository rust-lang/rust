// We disable the GC for this test because it would change what is printed. We are testing the
// printing, not how it interacts with the GC.
//@compile-flags: -Zmiri-permissive-provenance -Zmiri-provenance-gc=0

use std::alloc::{self, Layout};
use std::mem::ManuallyDrop;

extern "Rust" {
    fn miri_get_alloc_id(ptr: *const u8) -> u64;
    fn miri_print_borrow_state(alloc_id: u64, show_unnamed: bool);
}

fn get_alloc_id(ptr: *const u8) -> u64 {
    unsafe { miri_get_alloc_id(ptr) }
}

fn print_borrow_stacks(alloc_id: u64) {
    unsafe {
        miri_print_borrow_state(alloc_id, /* ignored: show_unnamed */ false)
    }
}

fn main() {
    let ptr = unsafe { alloc::alloc(Layout::new::<u8>()) };
    let alloc_id = get_alloc_id(ptr);
    print_borrow_stacks(alloc_id);

    assert!(!ptr.is_null());
    print_borrow_stacks(alloc_id);

    unsafe { *ptr = 42 };
    print_borrow_stacks(alloc_id);

    let _b = unsafe { ManuallyDrop::new(Box::from_raw(ptr)) };
    print_borrow_stacks(alloc_id);

    let _ptr = unsafe { &*ptr };
    print_borrow_stacks(alloc_id);

    // Create an unknown bottom, and print it
    let ptr = ptr as usize as *mut u8;
    unsafe {
        *ptr = 5;
    }
    print_borrow_stacks(alloc_id);

    unsafe { alloc::dealloc(ptr, Layout::new::<u8>()) };
}
