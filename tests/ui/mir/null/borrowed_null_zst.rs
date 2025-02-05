//@ run-fail
//@ compile-flags: -C debug-assertions
//@ error-pattern: null pointer dereference occured

fn main() {
    let ptr: *const () = std::ptr::null();
    let _ptr: &() = unsafe { &*ptr };
}
