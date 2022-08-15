// This tests demonstrates the effect of 'Disabling' mutable references on reads, rather than
// removing them from the stack -- the latter would 'merge' neighboring SRW groups which we would
// like to avoid.
fn main() {
    unsafe {
        let mut mem = 0;
        let base = &mut mem as *mut i32; // the base pointer we build the rest of the stack on
        let mutref = &mut *base;
        let raw = mutref as *mut i32;
        // in the stack, we now have [base, mutref, raw]
        let _val = *base;
        // now mutref is disabled
        *base = 1;
        // this should pop raw from the stack, since it is in a different SRW group
        let _val = *raw; //~ERROR: that tag does not exist in the borrow stack
    }
}
