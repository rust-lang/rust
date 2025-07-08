//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results

fn main() {
    let ptr: *mut u32 = std::ptr::null_mut();
    let _ptr: &mut u32 = unsafe { &mut *ptr };
}
