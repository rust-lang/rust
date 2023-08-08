// This should fail even without validation
//@compile-flags: -Zmiri-disable-validation -Cdebug-assertions=no

fn main() {
    // Try many times as this might work by chance.
    for i in 0..20 {
        let x = i as u8;
        let x = &x as *const _ as *const [u32; 0];
        // This must fail because alignment is violated. Test specifically for loading ZST.
        let _x = unsafe { *x }; //~ERROR: alignment 4 is required
    }
}
