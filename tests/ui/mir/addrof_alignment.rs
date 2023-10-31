// run-pass
// ignore-wasm32-bare: No panic messages
// compile-flags: -C debug-assertions

struct Misalignment {
    a: u32,
}

fn main() {
    let items: [Misalignment; 2] = [Misalignment { a: 0 }, Misalignment { a: 1 }];
    unsafe {
        let ptr: *const Misalignment = items.as_ptr().cast::<u8>().add(1).cast::<Misalignment>();
        let _ptr = core::ptr::addr_of!((*ptr).a);
    }
}
