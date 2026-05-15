#![deny(fuzzy_provenance_casts)]
#![deny(lossy_provenance_casts)]

use core::ffi::{c_intptr_t, c_uintptr_t};

#[test]
fn test_intptr_unitptr() {
    // These types should have the same size as a pointer.
    assert_eq!(core::mem::size_of::<c_intptr_t>(), core::mem::size_of::<*const ()>());
    assert_eq!(core::mem::size_of::<c_uintptr_t>(), core::mem::size_of::<*const ()>());

    let ptr = core::ptr::with_exposed_provenance(16_usize);
    let ptr_uintptr_t = c_uintptr_t(ptr);
    let ptr_back = ptr_uintptr_t.0;
    assert_eq!(ptr_back.addr(), 16_usize);
    assert_eq!(ptr, ptr_back);
}
