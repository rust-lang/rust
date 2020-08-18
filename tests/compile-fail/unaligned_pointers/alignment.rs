// error-pattern: but alignment 4 is required

fn main() {
    // No retry needed, this fails reliably.

    let mut x = [0u8; 20];
    let x_ptr: *mut u8 = x.as_mut_ptr();
    // At least one of these is definitely unaligned.
    unsafe {
        *(x_ptr as *mut u32) = 42;
        *(x_ptr.add(1) as *mut u32) = 42;
    }
}
