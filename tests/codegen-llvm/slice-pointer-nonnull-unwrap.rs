//@ compile-flags: -Copt-level=3
#![crate_type = "lib"]

use std::ptr::NonNull;

// CHECK-LABEL: @slice_ptr_len_1
// CHECK-NEXT: {{.*}}:
// CHECK-NEXT: ret {{i(32|64)}} %ptr.1
#[no_mangle]
pub fn slice_ptr_len_1(ptr: *const [u8]) -> usize {
    let ptr = ptr.cast_mut();
    if let Some(ptr) = NonNull::new(ptr) {
        ptr.len()
    } else {
        // We know ptr is null, so we know ptr.wrapping_byte_add(1) is not null.
        NonNull::new(ptr.wrapping_byte_add(1)).unwrap().len()
    }
}
