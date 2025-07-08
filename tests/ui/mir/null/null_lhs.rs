//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results

fn main() {
    let ptr: *mut u32 = std::ptr::null_mut();
    unsafe {
        *(ptr) = 42;
    }
}
