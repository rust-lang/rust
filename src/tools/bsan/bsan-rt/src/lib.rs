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
use core::ops::Deref;
#[cfg(not(test))]
use core::panic::PanicInfo;
use core::ptr;
use core::ptr::null;

use libc::glob;

mod global;
use global::{global_ctx, init_global_ctx};

mod bsan_alloc;
pub use bsan_alloc::BsanAllocator;
#[cfg(test)]
pub use bsan_alloc::TEST_ALLOC;

use crate::shadow::{L2, table_indices};

mod shadow;

type AllocID = usize;
type BorrowTag = usize;

#[derive(Debug)]
pub struct Provenance {
    lock_address: *mut c_void,
    alloc_id: AllocID,
    borrow_tag: BorrowTag,
}

impl Copy for Provenance {}
impl Clone for Provenance {
    fn clone(&self) -> Self {
        Provenance {
            lock_address: self.lock_address,
            alloc_id: self.alloc_id,
            borrow_tag: self.borrow_tag,
        }
    }
}
impl shadow::Provenance for Provenance {}

#[no_mangle]
unsafe extern "C" fn bsan_init(alloc: BsanAllocator) {
    init_global_ctx(alloc);
}

fn null_prov() -> Provenance {
    Provenance { lock_address: null() as *mut c_void, alloc_id: 0, borrow_tag: 0 }
}
#[no_mangle]
unsafe extern "C" fn bsan_load_prov(ptr: *mut c_void) -> Provenance {
    if ptr.is_null() {
        return null_prov();
    }
    let (l1_addr, l2_addr) = table_indices(ptr as usize);
    let l1 = &global_ctx().shadow_heap.l1;
    let mut l2 = l1.entries[l1_addr];
    if l2.is_null() {
        return null_prov();
    }

    let prov = l2.lookup_mut(l2_addr);

    *prov
}

#[no_mangle]
unsafe extern "C" fn bsan_store_prov(provenance: *const Provenance) {
    if provenance.is_null() {
        return;
    }
    let (l1_addr, l2_addr) = table_indices(provenance.lock_address as usize);
    let l1 = &global_ctx().shadow_heap.l1;
    let mut l2 = l1.entries[l1_addr];
    if l2.is_null() {
        l2 = &mut L2::new(global_ctx().allocator);
        l1.entries[l1_addr] = l2;
    }

    l2.bytes[l2_addr] = *provenance;
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
