//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: null pointer dereference occurred

fn main() {
    let ptr: *const () = std::ptr::null();
    let _ptr: &() = unsafe { &*ptr };
}
