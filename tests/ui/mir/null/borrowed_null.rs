//@ run-fail
//@ compile-flags: -C debug-assertions
//@ error-pattern: null pointer dereference occured

fn main() {
    let ptr: *const u32 = std::ptr::null();
    let _ptr: &u32 = unsafe { &*ptr };
}
