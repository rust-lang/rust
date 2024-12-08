// Should be caught even without retagging
//@compile-flags: -Zmiri-disable-stacked-borrows
use std::ptr::{self, addr_of_mut};

// Deref'ing a dangling raw pointer is fine, but for a dangling reference it is not.
// We do this behind a pointer indirection to potentially fool validity checking.
// (This test relies on the `deref_copy` pass that lowers `**ptr` to materialize the intermediate pointer.)

fn main() {
    let mut inner = ptr::without_provenance::<i32>(24);
    let outer = addr_of_mut!(inner).cast::<&'static mut i32>();
    // Now `outer` is a pointer to a dangling reference.
    // Deref'ing that should be UB.
    let _val = unsafe { addr_of_mut!(**outer) }; //~ERROR: dangling reference
}
