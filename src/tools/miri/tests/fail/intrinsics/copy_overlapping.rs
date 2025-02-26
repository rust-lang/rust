#![feature(intrinsics)]

// Directly call intrinsic to avoid debug assertions in libstd
#[rustc_intrinsic]
unsafe fn copy_nonoverlapping<T>(_src: *const T, _dst: *mut T, _count: usize);

fn main() {
    let mut data = [0u8; 16];
    unsafe {
        let a = data.as_mut_ptr();
        let b = a.wrapping_offset(1) as *mut _;
        copy_nonoverlapping(a, b, 2); //~ ERROR: `copy_nonoverlapping` called on overlapping ranges
    }
}
