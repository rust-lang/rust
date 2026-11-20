//@compile-flags: -Zmiri-tree-borrows

// Read from local variable kills reborrows *and* raw pointers derived from them.
// This test would be accepted under Stacked Borrows.
fn main() {
    unsafe {
        let mut root = 6u8;
        let mref = &mut root;
        let ptr = mref as *mut u8;
        *ptr = 0; // Write
        assert_eq!(root, 0); // Parent Read
        *ptr = 0; //~ ERROR: /write access through .* is forbidden/
    }
}
