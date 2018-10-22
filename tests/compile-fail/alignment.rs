fn main() {
    // miri always gives allocations the worst possible alignment, so a `u8` array is guaranteed
    // to be at the virtual location 1 (so one byte offset from the ultimate alignemnt location 0)
    let mut x = [0u8; 20];
    let x_ptr: *mut u8 = &mut x[0];
    let y_ptr = x_ptr as *mut u64;
    unsafe {
        *y_ptr = 42; //~ ERROR tried to access memory with alignment 1, but alignment
    }
    panic!("unreachable in miri");
}
