#![feature(intrinsics)]

// Directly call intrinsic to avoid debug assertions in libstd
extern "rust-intrinsic" {
    fn copy_nonoverlapping<T>(src: *const T, dst: *mut T, count: usize);
}

fn main() {
    let mut data = [0u8; 16];
    unsafe {
        let a = data.as_mut_ptr();
        let b = a.wrapping_offset(1) as *mut _;
        copy_nonoverlapping(a, b, 2); //~ ERROR: `copy_nonoverlapping` called on overlapping ranges
    }
}
