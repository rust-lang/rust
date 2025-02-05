#![no_std]
#![feature(strict_overflow_ops)]
#![allow(unused)]

extern crate alloc;

mod allocator;
use allocator::BsanAlloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: BsanAlloc = BsanAlloc {};

mod shadow;

use core::ffi::c_void;
use core::num::NonZero;

use log::info;

#[no_mangle]
extern "C" fn bsan_init() {
    let _ = env_logger::builder().try_init();
    info!("Initialized global state");
}

#[no_mangle]
extern "C" fn bsan_expose_tag(ptr: *mut c_void) {
    info!("Exposed tag for pointer: {:?}", ptr);
}

#[no_mangle]
extern "C" fn bsan_retag(ptr: *mut c_void, retag_kind: u8, place_kind: u8) -> u64 {
    info!("Retagged pointer: {:?}", ptr);
    0
}

#[no_mangle]
extern "C" fn bsan_read(ptr: *mut c_void, access_size: u64) {
    info!("Reading {} bytes starting at address: {:?}", access_size, ptr);
}

#[no_mangle]
extern "C" fn bsan_write(ptr: *mut c_void, access_size: u64) {
    info!("Writing {} bytes starting at address: {:?}", access_size, ptr);
}

#[no_mangle]
extern "C" fn bsan_func_entry() {
    info!("Entered function");
}

#[no_mangle]
extern "C" fn bsan_func_exit() {
    info!("Exited function");
}
