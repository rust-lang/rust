//@ run-fail
//@ compile-flags: -C debug-assertions
//@ check-run-results

fn main() {
    let ptr: *const () = std::ptr::null();
    let _ptr: &() = unsafe { &*ptr };
}
