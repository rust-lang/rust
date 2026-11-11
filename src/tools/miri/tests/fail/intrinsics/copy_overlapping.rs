#![feature(core_intrinsics)]

fn main() {
    let mut data = [0u8; 16];
    unsafe {
        let a = data.as_mut_ptr();
        let b = a.wrapping_offset(1) as *mut _;
        // Directly call intrinsic to avoid debug assertions in the `std::ptr` version.
        std::intrinsics::copy_nonoverlapping(a, b, 2); //~ ERROR: `copy_nonoverlapping` called on overlapping ranges
    }
}
