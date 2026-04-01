// Make sure that we don't insert a check for zero-sized reads or writes to
// null, because they are allowed.
//@ run-pass
//@ compile-flags: -C debug-assertions

fn main() {
    let ptr: *mut () = std::ptr::null_mut();
    unsafe {
        *(ptr) = ();
    }
    let ptr1: *const () = std::ptr::null_mut();
    unsafe {
        let _ptr = *ptr1;
    }
}
