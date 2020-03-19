// This should fail even without validation
// compile-flags: -Zmiri-disable-validation

fn main() {
    let x = &2u16;
    let x = x as *const _ as *const [u32; 0];
    // This must fail because alignment is violated. Test specifically for loading ZST.
    let _x = unsafe { *x };
    //~^ ERROR memory with alignment 2, but alignment 4 is required
}
