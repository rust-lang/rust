fn main() {
    // Not using the () type here, as writes of that type do not even have MIR generated.
    // Also not assigning directly as that's array initialization, not assignment.
    let zst_val = [1u8; 0];

    // make sure ZST accesses are checked against being "truly" dangling pointers
    // (into deallocated allocations).
    let mut x_box = Box::new(1u8);
    let x = &mut *x_box as *mut _ as *mut [u8; 0];
    drop(x_box);
    unsafe { *x = zst_val }; //~ ERROR: dereferenced after this allocation got freed
}
