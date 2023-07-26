//@compile-flags: -Zmiri-tree-borrows

// Protectors should not trigger on uninitialized locations,
// otherwise we can write safe code that triggers UB.
// To test this we do a write access that disables a protected
// location, but the location is actually outside of the protected range.

// Both x and y are protected here
fn write_second(_x: &mut u8, y: &mut u8) {
    *y = 1;
}

fn main() {
    let mut data = (0u8, 1u8);
    write_second(&mut data.0, &mut data.1);
}
