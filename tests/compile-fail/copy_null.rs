//error-pattern: invalid use of NULL pointer

fn main() {
    let mut data = [0u16; 4];
    let ptr = &mut data[0] as *mut u16;
    // Even copying 0 elements from NULL should error.
    unsafe { ptr.copy_from(std::ptr::null(), 0); }
}
