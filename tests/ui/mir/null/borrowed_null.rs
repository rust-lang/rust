//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results

fn main() {
    let ptr: *const u32 = std::ptr::null();
    let _ptr: &u32 = unsafe { &*ptr };
}
