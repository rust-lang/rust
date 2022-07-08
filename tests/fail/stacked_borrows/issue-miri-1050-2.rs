//@error-pattern: is a dangling pointer
use std::ptr::NonNull;

fn main() {
    unsafe {
        let ptr = NonNull::<i32>::dangling();
        Box::from_raw(ptr.as_ptr());
    }
}
