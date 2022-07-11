// Make sure we find these even with many checks disabled.
//@compile-flags: -Zmiri-disable-alignment-check -Zmiri-disable-stacked-borrows -Zmiri-disable-validation

// Test what happens when we overwrite parts of a pointer.
// Also see <https://github.com/rust-lang/miri/issues/2181>.

fn main() {
    let mut p = &42;
    unsafe {
        let ptr: *mut _ = &mut p;
        *(ptr as *mut u8) = 123; // if we ever support 8 bit pointers, this is gonna cause
        // "attempted to interpret some raw bytes as a pointer address" instead of
        // "attempted to read undefined bytes"
    }
    let x = *p; //~ ERROR: this operation requires initialized memory
    panic!("this should never print: {}", x);
}
