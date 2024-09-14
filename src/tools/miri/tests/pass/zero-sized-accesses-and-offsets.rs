//! Tests specific for <https://github.com/rust-lang/rust/issues/117945>: zero-sized operations.

use std::ptr;

fn main() {
    // Null.
    test_ptr(ptr::null_mut::<()>());
    // No provenance.
    test_ptr(ptr::without_provenance_mut::<()>(1));
    // Out-of-bounds.
    let mut b = Box::new(0i32);
    let ptr = ptr::addr_of_mut!(*b) as *mut ();
    test_ptr(ptr.wrapping_byte_add(2));
    // Dangling (use-after-free).
    drop(b);
    test_ptr(ptr);
}

fn test_ptr(ptr: *mut ()) {
    unsafe {
        // Reads and writes.
        let mut val = *ptr;
        *ptr = val;
        ptr.read();
        ptr.write(());
        // Memory access intrinsics.
        // - memcpy (1st and 2nd argument)
        ptr.copy_from_nonoverlapping(&(), 1);
        ptr.copy_to_nonoverlapping(&mut val, 1);
        // - memmove (1st and 2nd argument)
        ptr.copy_from(&(), 1);
        ptr.copy_to(&mut val, 1);
        // - memset
        ptr.write_bytes(0u8, 1);
        // Offset.
        let _ = ptr.offset(0);
        let _ = ptr.offset(1); // this is still 0 bytes
        // Distance.
        let ptr = ptr.cast::<i32>();
        ptr.offset_from(ptr);
        // Distance from other "bad" pointers that have the same address, but different provenance. Some
        // of this is library UB, but we don't want it to be language UB since that would violate
        // provenance monotonicity: if we allow computing the distance between two ptrs with no
        // provenance, we have to allow computing it between two ptrs with arbitrary provenance.
        // - Distance from "no provenance"
        ptr.offset_from(ptr::without_provenance_mut(ptr.addr()));
        // - Distance from out-of-bounds pointer
        let mut b = Box::new(0i32);
        let other_ptr = ptr::addr_of_mut!(*b);
        ptr.offset_from(other_ptr.with_addr(ptr.addr()));
        // - Distance from use-after-free pointer
        drop(b);
        ptr.offset_from(other_ptr.with_addr(ptr.addr()));
    }
}
