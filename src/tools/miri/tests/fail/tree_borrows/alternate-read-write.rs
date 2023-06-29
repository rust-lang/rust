//@compile-flags: -Zmiri-tree-borrows

// Check that TB properly rejects alternating Reads and Writes, but tolerates
// alternating only Reads to Reserved mutable references.
pub fn main() {
    let x = &mut 0u8;
    let y = unsafe { &mut *(x as *mut u8) };
    // Foreign Read, but this is a no-op from the point of view of y (still Reserved)
    let _val = *x;
    // Now we activate y, for this to succeed y needs to not have been Frozen
    // by the previous operation
    *y += 1; // Success
    // This time y gets Frozen...
    let _val = *x;
    // ... and the next Write attempt fails.
    *y += 1; // Failure //~ ERROR: /write access through .* is forbidden/
    let _val = *x;
    *y += 1; // Unreachable
}
