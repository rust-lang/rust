// Make sure that we don't insert a check for `addr_of!`.
//@ run-pass
//@ compile-flags: -C debug-assertions

struct Field {
    a: u32,
}

fn main() {
    unsafe {
        let ptr: *const Field = std::ptr::null();
        let _ptr = core::ptr::addr_of!((*ptr).a);
    }
}
