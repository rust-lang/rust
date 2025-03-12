#![cfg(code)]

use crate::collect;

#[inline(never)]
#[no_mangle]
unsafe fn dummy() {
    if *(5345 as *const bool) {
        panic!("whoops")
    }
}

#[no_mangle]
unsafe fn pin_test() {
    collect::pin(|_| dummy());
}

#[no_mangle]
unsafe fn collect_test() {
    collect::collect();
}

#[no_mangle]
unsafe fn release_test() {
    collect::release();
}
