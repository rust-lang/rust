#![feature(core_intrinsics)]

//error-pattern: copy_nonoverlapping called on overlapping ranges

fn main() {
    let mut data = [0u8; 16];
    unsafe {
        let a = data.as_mut_ptr();
        let b = a.wrapping_offset(1) as *mut _;
        std::ptr::copy_nonoverlapping(a, b, 2);
    }
}
