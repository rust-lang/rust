// Some optimizations remove ZST accesses, thus masking this UB.
// compile-flags: -Zmir-opt-level=0

fn main() {
    // Not using the () type here, as writes of that type do not even have MIR generated.
    // Also not assigning directly as that's array initialization, not assignment.
    let zst_val = [1u8; 0];
    unsafe { *std::ptr::null_mut() = zst_val }; //~ ERROR memory access failed: 0x0 is not a valid pointer
}
