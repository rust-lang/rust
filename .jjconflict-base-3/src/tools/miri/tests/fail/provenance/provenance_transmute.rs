//@compile-flags: -Zmiri-permissive-provenance

#![allow(integer_to_ptr_transmutes)]

use std::mem;

// This is the example from
// <https://github.com/rust-lang/unsafe-code-guidelines/issues/286#issuecomment-1085144431>.

unsafe fn deref(left: *const u8, right: *const u8) {
    let left_int: usize = mem::transmute(left);
    let right_int: usize = mem::transmute(right);
    if left_int == right_int {
        // The compiler is allowed to replace `left_int` by `right_int` here...
        let left_ptr: *const u8 = mem::transmute(left_int);
        // ...which however means here it could be dereferencing the wrong pointer.
        let _val = *left_ptr; //~ERROR: dangling pointer
    }
}

fn main() {
    let ptr1 = &0u8 as *const u8;
    let ptr2 = &1u8 as *const u8;
    unsafe {
        // Two pointers with the same address but different provenance.
        deref(ptr1, ptr2.with_addr(ptr1.addr()));
    }
}
