//@compile-flags: -Zmiri-tree-borrows -Zmiri-provenance-gc=0

// Shows the effect of the optimization of #4008.
// The diagnostics change, but not the error itself.

// When this method is called, the tree will be a single line and look like this,
// with other_ptr being the root at the top
// other_ptr = root : Active
// intermediary     : Frozen // an intermediary node
// m                : Reserved
fn write_to_mut(m: &mut u8, other_ptr: *const u8) {
    unsafe {
        std::hint::black_box(*other_ptr);
    }
    // In line 17 above, m should have become Reserved (protected) so that this write is impossible.
    // However, that does not happen because the read above is not forwarded to the subtree below
    // the Frozen intermediary node. This does not affect UB, however, because the Frozen that blocked
    // the read already prevents any child writes.
    *m = 42; //~ERROR: /write access through .* is forbidden/
}

fn main() {
    let root = 42u8;
    unsafe {
        let intermediary = &root;
        let data = &mut *(core::ptr::addr_of!(*intermediary) as *mut u8);
        write_to_mut(data, core::ptr::addr_of!(root));
    }
}
