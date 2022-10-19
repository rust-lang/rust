use std::{
    alloc::{self, Layout},
    mem::ManuallyDrop,
};

extern "Rust" {
    fn miri_get_alloc_id(ptr: *const u8) -> u64;
    fn miri_print_stacks(alloc_id: u64);
}

fn main() {
    let ptr = unsafe { alloc::alloc(Layout::new::<u8>()) };
    let alloc_id = unsafe { miri_get_alloc_id(ptr) };
    unsafe { miri_print_stacks(alloc_id) };

    assert!(!ptr.is_null());
    unsafe { miri_print_stacks(alloc_id) };

    unsafe { *ptr = 42 };
    unsafe { miri_print_stacks(alloc_id) };

    let _b = unsafe { ManuallyDrop::new(Box::from_raw(ptr)) };
    unsafe { miri_print_stacks(alloc_id) };

    let _ptr = unsafe { &*ptr };
    unsafe { miri_print_stacks(alloc_id) };

    unsafe { alloc::dealloc(ptr, Layout::new::<u8>()) };
}
