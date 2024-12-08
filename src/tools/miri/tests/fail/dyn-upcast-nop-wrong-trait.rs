// This upcast is currently forbidden because it involves an invalid value.
// However, if in the future we relax the validity requirements for raw pointer vtables,
// we could consider allowing this again -- the cast itself isn't doing anything wrong,
// only the transmutes needed to set up the testcase are wrong.

use std::fmt;

fn main() {
    // vtable_mismatch_nop_cast
    let ptr: &dyn fmt::Display = &0;
    let ptr: *const (dyn fmt::Debug + Send + Sync) = unsafe { std::mem::transmute(ptr) }; //~ERROR: wrong trait
    // Even though the vtable is for the wrong trait, this cast doesn't actually change the needed
    // vtable so it should still be allowed -- if we ever allow the line above.
    let _ptr2 = ptr as *const dyn fmt::Debug;
}
