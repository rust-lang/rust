// This tests demonstrates the effect of 'Disabling' mutable references on reads, rather than
// removing them from the stack -- the latter would 'merge' neighboring SRW groups which we would
// like to avoid.
fn main() {
    unsafe {
        let mut mem = 0;
        let base = &mut mem as *mut i32; // the base pointer we build the rest of the stack on
        let raw = {
            let mutref = &mut *base;
            mutref as *mut i32
        };
        // In the stack, we now have [base, mutref, raw].
        // We do this in a weird way where `mutref` is out of scope here, just in case
        // Miri decides to get smart and argue that items for tags that are no longer
        // used by any pointer stored anywhere in the machine can be removed.
        let _val = *base;
        // now mutref is disabled
        *base = 1;
        // this should pop raw from the stack, since it is in a different SRW group
        let _val = *raw; //~ERROR: that tag does not exist in the borrow stack
    }
}
