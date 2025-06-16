#![feature(rustc_attrs)]
#![no_std]
#![no_main]

extern crate alloc;

use alloc::alloc::{GlobalAlloc, Layout};

#[panic_handler]
fn panic_handler(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

#[no_mangle]
extern "C" fn rust_eh_personality() {
    loop {}
}

#[global_allocator]
static ALLOC: Alloc = Alloc;

struct Alloc;

unsafe impl GlobalAlloc for Alloc {
    unsafe fn alloc(&self, _: Layout) -> *mut u8 {
        core::ptr::null_mut()
    }
    unsafe fn dealloc(&self, _: *mut u8, _: Layout) {
        todo!()
    }
}

#[cfg(not(check_feature_gate))]
#[rustc_std_internal_symbol]
fn __rust_no_alloc_shim_is_unstable_v2() {}

#[no_mangle]
extern "C" fn main(_argc: core::ffi::c_int, _argv: *const *const i8) -> i32 {
    unsafe {
        assert_eq!(alloc::alloc::alloc(Layout::new::<()>()), core::ptr::null_mut());
    }

    0
}
