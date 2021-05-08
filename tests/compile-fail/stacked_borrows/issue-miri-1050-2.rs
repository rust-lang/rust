// error-pattern: 0x4 is not a valid pointer
use std::ptr::NonNull;

fn main() { unsafe {
    let ptr = NonNull::<i32>::dangling();
    Box::from_raw(ptr.as_ptr());
} }
