fn main() {
    // Not using the () type here, as writes of that type do not even have MIR generated.
    // Also not assigning directly as that's array initialization, not assignment.
    let zst_val = [1u8; 0];

    // make sure ZST accesses are checked against being "truly" dangling pointers
    // (that are out-of-bounds).
    let mut x_box = Box::new(1u8);
    let x = (&mut *x_box as *mut u8).wrapping_offset(1);
    // This one is just "at the edge", but still okay
    unsafe { *(x as *mut [u8; 0]) = zst_val };
    // One byte further is OOB.
    let x = x.wrapping_offset(1);
    unsafe { *(x as *mut [u8; 0]) = zst_val }; //~ ERROR: out-of-bounds
}
