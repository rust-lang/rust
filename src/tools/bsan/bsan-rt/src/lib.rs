#![cfg_attr(not(test), no_std)]
#![feature(allocator_api)]
#![feature(sync_unsafe_cell)]
#![feature(alloc_layout_extra)]
#![feature(strict_overflow_ops)]
#![allow(unused)]

extern crate alloc;

use core::cell::UnsafeCell;
use core::ffi::c_void;
use core::num::NonZero;
#[cfg(not(test))]
use core::panic::PanicInfo;

mod global;
use global::{global_ctx, init_global_ctx};

mod bsan_alloc;
pub use bsan_alloc::BsanAllocator;
#[cfg(test)]
pub use bsan_alloc::TEST_ALLOC;

mod shadow;

type AllocID = usize;
type BorrowTag = usize;

#
pub struct Provenance {
    lock_address : *mut c_void,
    alloc_id : AllocID,
    borrow_tag: BorrowTag,
}

impl 

#[no_mangle]
unsafe extern "C" fn bsan_init(alloc: BsanAllocator) {
    init_global_ctx(alloc);
}

#[no_mangle]
extern "C" fn bsan_load_prov(ptr: *mut c_void) -> Provenance { // TODO implement null function (cannot use options)
    // TODO: get the global context, and then through their call a shadow heap method/function
    return Provenance {}
}

#[no_mangle]
extern "C" fn bsan_store_prov(provenance : *const Provenance) {
    // TODO: store the provenance in the shadow heap, and then through a call to the shadow heap store it
}

#[no_mangle]
extern "C" fn bsan_expose_tag(ptr: *mut c_void) {}

#[no_mangle]
extern "C" fn bsan_retag(ptr: *mut c_void, retag_kind: u8, place_kind: u8) -> u64 {
    0
}

#[no_mangle]
extern "C" fn bsan_read(ptr: *mut c_void, access_size: u64) {}

#[no_mangle]
extern "C" fn bsan_write(ptr: *mut c_void, access_size: u64) {}

#[no_mangle]
extern "C" fn bsan_func_entry() {}

#[no_mangle]
extern "C" fn bsan_func_exit() {}

#[cfg(not(test))]
#[panic_handler]
fn panic(info: &PanicInfo<'_>) -> ! {
    loop {}
}
