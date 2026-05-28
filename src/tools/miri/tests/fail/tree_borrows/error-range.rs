//@compile-flags: -Zmiri-tree-borrows

use core::ptr::addr_of_mut;

// Check that the diagnostics correctly report the exact range at fault
// and trim irrelevant events.
fn main() {
    unsafe {
        let data = &mut [0u8; 16];

        // Create and activate a few bytes
        let rmut = &mut *addr_of_mut!(data[0..6]);
        rmut[3] += 1;
        rmut[4] += 1;
        rmut[5] += 1;

        // Now make them lose some perms
        let _v = data[3];
        let _v = data[4];
        let _v = data[5];
        data[4] = 1;
        data[5] = 1;

        // Final test
        rmut[5] += 1; //~ ERROR: /read access through .* is forbidden/
    }
}
