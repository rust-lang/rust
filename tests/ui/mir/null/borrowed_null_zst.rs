//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: null reference produced

fn main() {
    let ptr: *const () = std::ptr::null();
    let _ptr: &() = unsafe { &*ptr };
}
