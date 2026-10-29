//@compile-flags: -Zmiri-disable-validation -Zmiri-ignore-leaks
// This used to cause ICEs in the aliasing model before validation existed. Now that aliasing checks
// are only triggered by validation, that configuration is no longer possible but we keep the test
// around just in case.

fn main() {
    unsafe {
        let ptr = Box::into_raw(Box::new(0u16));
        std::mem::forget(Box::from_raw(ptr as *mut u32));
    }
}
