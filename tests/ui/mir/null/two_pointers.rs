//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: null pointer dereference occurred

fn main() {
    let ptr = std::ptr::null();
    let mut dest = 0u32;
    let dest_ptr = &mut dest as *mut u32;
    unsafe {
        *dest_ptr = *(ptr);
    }
}
