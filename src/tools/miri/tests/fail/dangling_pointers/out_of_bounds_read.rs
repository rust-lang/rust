fn main() {
    let v: Vec<u16> = vec![1, 2];
    // This read is also misaligned. We make sure that the OOB message has priority.
    let x = unsafe { *v.as_ptr().wrapping_byte_add(5) }; //~ ERROR: out-of-bounds
    panic!("this should never print: {}", x);
}
