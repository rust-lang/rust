#![feature(core_intrinsics)]

fn main() {
    let mut a = [0u32; 2];
    let p = unsafe { a.as_mut_ptr().byte_add(1) };
    unsafe { std::intrinsics::write_bytes(p, 0, 1) };
    //~^ERROR: alignment 4 is required
}
