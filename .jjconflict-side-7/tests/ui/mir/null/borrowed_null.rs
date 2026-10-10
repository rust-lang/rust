//@ run-crash
//@ compile-flags: -C debug-assertions
//@ error-pattern: null reference produced

fn main() {
    let ptr: *const u32 = std::ptr::null();
    let _ptr: &u32 = unsafe { &*ptr };
}
