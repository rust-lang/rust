// This crate is intentionally empty and a re-export of `rustc_driver_impl` to allow the code in
// `rustc_driver_impl` to be compiled in parallel with other crates.

use std::os::raw::{c_char, c_int, c_void};

pub use rustc_driver_impl::*;

#[used]
static _F1: unsafe extern "C" fn(usize, usize) -> *mut c_void = calloc;
#[used]
static _F2: unsafe extern "C" fn(*mut *mut c_void, usize, usize) -> c_int = posix_memalign;
#[used]
static _F3: unsafe extern "C" fn(usize, usize) -> *mut c_void = aligned_alloc;
#[used]
static _F4: unsafe extern "C" fn(usize) -> *mut c_void = malloc;
#[used]
static _F5: unsafe extern "C" fn(*mut c_void, usize) -> *mut c_void = realloc;
#[used]
static _F6: unsafe extern "C" fn(*mut c_void) = free;
#[used]
static _F7: unsafe extern "C" fn(*const c_char) -> *mut c_char = strdup;

#[unsafe(no_mangle)]
unsafe extern "C" fn calloc(items: usize, size: usize) -> *mut c_void {
    unsafe { fjall::c::calloc(items, size) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn posix_memalign(ptr: *mut *mut c_void, size: usize, align: usize) -> c_int {
    unsafe { fjall::c::posix_memalign(ptr, size, align) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn aligned_alloc(size: usize, align: usize) -> *mut c_void {
    fjall::c::aligned_alloc(size, align)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn malloc(size: usize) -> *mut c_void {
    fjall::c::malloc(size)
}

#[unsafe(no_mangle)]
unsafe extern "C" fn realloc(ptr: *mut c_void, size: usize) -> *mut c_void {
    unsafe { fjall::c::realloc(ptr, size) }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn free(ptr: *mut c_void) {
    unsafe {
        fjall::c::free(ptr);
    }
}

#[unsafe(no_mangle)]
unsafe extern "C" fn strdup(ptr: *const c_char) -> *mut c_char {
    unsafe { fjall::c::strdup(ptr) }
}
