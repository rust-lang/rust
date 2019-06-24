// This should fail even without validation
// compile-flags: -Zmiri-disable-validation

fn main() {
    let x = [2u16, 3, 4]; // Make it big enough so we don't get an out-of-bounds error.
    let x = &x[0] as *const _ as *const u32;
    // This must fail because alignment is violated: the allocation's base is not sufficiently aligned.
    let _x = unsafe { *x }; //~ ERROR tried to access memory with alignment 2, but alignment 4 is required
}
