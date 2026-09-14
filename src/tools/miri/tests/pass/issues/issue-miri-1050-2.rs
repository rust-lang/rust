//@compile-flags: -Zmiri-disable-validation
// This used to cause ICEs in the aliasing model before validation existed. Now that aliasing checks
// are only triggered by validation, that configuration is no longer possible but we keep the test
// around just in case.
use std::ptr::NonNull;

fn main() {
    unsafe {
        let ptr = NonNull::<i32>::dangling();
        std::mem::forget(Box::from_raw(ptr.as_ptr()));
    }
}
