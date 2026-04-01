//@revisions: stack tree
// Ensure this even hits the aliasing model
//@compile-flags: -Zmiri-disable-validation
//@[tree]compile-flags: -Zmiri-tree-borrows
//@error-in-other-file: is a dangling pointer
use std::ptr::NonNull;

fn main() {
    unsafe {
        let ptr = NonNull::<i32>::dangling();
        drop(Box::from_raw(ptr.as_ptr()));
    }
}
