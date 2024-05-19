//@ run-pass
//@ compile-flags: -C debug-assertions

struct Misalignment {
    a: u32,
}

fn main() {
    let items: [Misalignment; 2] = [Misalignment { a: 0 }, Misalignment { a: 1 }];
    unsafe {
        let ptr: *const Misalignment = items.as_ptr().byte_add(1);
        let _ptr = core::ptr::addr_of!((*ptr).a);
    }
}
