// no-prefer-dynamic

#![feature(allocator, core_intrinsics, panic_unwind)]
#![allocator]
#![crate_type = "rlib"]
#![no_std]

extern crate unwind;

pub static mut HITS: usize = 0;

type size_t = usize;

extern {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    fn calloc(size: usize, amt: usize) -> *mut u8;
    fn realloc(ptr: *mut u8, size: usize) -> *mut u8;
}

#[no_mangle]
pub extern fn __rust_allocate(size: usize, align: usize) -> *mut u8 {
    unsafe {
        HITS += 1;
        malloc(size as size_t) as *mut u8
    }
}

#[no_mangle]
pub extern fn __rust_allocate_zeroed(size: usize, _align: usize) -> *mut u8 {
    unsafe { calloc(size as size_t, 1) as *mut u8 }
}

#[no_mangle]
pub extern fn __rust_deallocate(ptr: *mut u8, old_size: usize, align: usize) {
    unsafe {
        HITS += 1;
        free(ptr as *mut _)
    }
}

#[no_mangle]
pub extern fn __rust_reallocate(ptr: *mut u8, old_size: usize, size: usize,
                                align: usize) -> *mut u8 {
    unsafe {
        realloc(ptr as *mut _, size as size_t) as *mut u8
    }
}

#[no_mangle]
pub extern fn __rust_reallocate_inplace(ptr: *mut u8, old_size: usize,
                                        size: usize, align: usize) -> usize {
    unsafe { core::intrinsics::abort() }
}

#[no_mangle]
pub extern fn __rust_usable_size(size: usize, align: usize) -> usize {
    unsafe { core::intrinsics::abort() }
}
