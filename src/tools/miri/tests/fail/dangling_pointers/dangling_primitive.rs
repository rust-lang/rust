// The interpreter tries to delay allocating locals until their address is taken.
// This test checks that we correctly use the span associated with the local itself, not the span
// where we take the address of the local and force it to be allocated.

fn main() {
    let ptr = {
        let x = 0usize; // This line should appear in the helps
        &x as *const usize // This line should NOT appear in the helps
    };
    unsafe {
        dbg!(*ptr); //~ ERROR: has been freed
    }
}
