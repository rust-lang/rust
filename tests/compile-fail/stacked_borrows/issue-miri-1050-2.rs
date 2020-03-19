// error-pattern: invalid use of 4 as a pointer
use std::ptr::NonNull;

fn main() { unsafe {
    let ptr = NonNull::<i32>::dangling();
    Box::from_raw(ptr.as_ptr());
} }
