// This should fail even without validation or Stacked Borrows.
//@compile-flags: -Zmiri-disable-validation -Zmiri-disable-stacked-borrows -Cdebug-assertions=no

fn main() {
    // No retry needed, this fails reliably.

    let x = [2u32, 3]; // Make it big enough so we don't get an out-of-bounds error.
    let x = (x.as_ptr() as *const u8).wrapping_offset(3) as *const u32;
    // This must fail because alignment is violated: the offset is not sufficiently aligned.
    // Also make the offset not a power of 2, that used to ICE.
    let _x = unsafe { *x }; //~ERROR: with alignment 1, but alignment 4 is required
}
