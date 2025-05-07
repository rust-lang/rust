//@ run-pass
//@ compile-flags: -C debug-assertions

#[repr(align(8))]
struct Misalignment {
    a: u8,
}

fn main() {
    let mem = 0u64;
    let ptr = &mem as *const u64 as *const Misalignment;
    unsafe {
        let ptr = ptr.byte_add(1);
        let _ref: &u8 = &(*ptr).a;
    }
}
