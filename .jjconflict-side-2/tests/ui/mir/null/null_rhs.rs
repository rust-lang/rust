//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: null pointer dereference occurred

fn main() {
    let ptr: *mut u32 = std::ptr::null_mut();
    unsafe {
        let _v = *ptr;
    }
}
