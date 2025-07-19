//@ run-crash
//@ compile-flags: -Cdebug-assertions=yes
//@ error-pattern: unsafe precondition(s) violated: Vec::from_raw_parts_in requires that length <= capacity
//@ revisions: vec_from_raw_parts vec_from_raw_parts_in string_from_raw_parts

#![feature(allocator_api)]

fn main() {
    let ptr = std::ptr::null_mut::<u8>();
    // Test Vec::from_raw_parts with length > capacity
    unsafe {
        #[cfg(vec_from_raw_parts)]
        let _vec = Vec::from_raw_parts(ptr, 10, 5);
    }

    // Test Vec::from_raw_parts_in with length > capacity
    unsafe {
        let alloc = std::alloc::Global;
        #[cfg(vec_from_raw_parts_in)]
        let _vec = Vec::from_raw_parts_in(ptr, 10, 5, alloc);
    }

    // Test String::from_raw_parts with length > capacity
    // Because it calls Vec::from_raw_parts, it should also fail
    unsafe {
        #[cfg(string_from_raw_parts)]
        let _vec = String::from_raw_parts(ptr, 10, 5);
    }
}
