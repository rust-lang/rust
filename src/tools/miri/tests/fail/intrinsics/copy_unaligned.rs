#![feature(core_intrinsics)]

fn main() {
    let mut data = [0u16; 8];
    let ptr = (&mut data[0] as *mut u16 as *mut u8).wrapping_add(1) as *mut u16;
    // Even copying 0 elements to something unaligned should error
    unsafe {
        // Directly call intrinsic to avoid debug assertions in the `std::ptr` version.
        std::intrinsics::copy_nonoverlapping(&data[5], ptr, 0); //~ ERROR: accessing memory with alignment 1, but alignment 2 is required
    }
}
