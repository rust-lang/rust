#![feature(pointer_byte_offsets)]

fn main() {
    let mut v: Vec<u16> = vec![1, 2];
    // This read is also misaligned. We make sure that the OOB message has priority.
    unsafe { *v.as_mut_ptr().wrapping_byte_add(5) = 0 }; //~ ERROR: out-of-bounds
}
