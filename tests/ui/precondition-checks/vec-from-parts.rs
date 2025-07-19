//@ run-crash
//@ compile-flags: -Cdebug-assertions=yes
//@ error-pattern: unsafe precondition(s) violated: Vec::from_parts_in requires that length <= capacity
#![feature(allocator_api)]

use std::ptr::NonNull;

fn main() {
    let ptr: NonNull<i32> = std::ptr::NonNull::dangling();
    // Test Vec::from_parts_in with length > capacity
    unsafe {
        let alloc = std::alloc::Global;
        let _vec = Vec::from_parts_in(ptr, 10, 5, alloc);
    }
}
