// Make sure that we don't insert a check for places that do not read.
//@ run-pass
//@ compile-flags: -C debug-assertions

fn main() {
    let ptr: *const () = std::ptr::null();
    unsafe {
        let _ = *ptr;
        let _ = &raw const *ptr;
    }
}
