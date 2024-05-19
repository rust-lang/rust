fn main() {
    // This pointer *could* be NULL so we cannot load from it, not even at ZST.
    // Not using the () type here, as writes of that type do not even have MIR generated.
    // Also not assigning directly as that's array initialization, not assignment.
    let zst_val = [1u8; 0];
    let ptr = (&0u8 as *const u8).wrapping_sub(0x800) as *mut [u8; 0];
    unsafe { *ptr = zst_val }; //~ ERROR: out-of-bounds
}
