//@ignore-target-windows: No libc on Windows
//@compile-flags: -Zmiri-permissive-provenance
// C's memcpy is 0 bytes is UB for some pointers that are allowed in Rust's `copy_nonoverlapping`.

fn main() {
    let from = 42 as *const u8;
    let to = 23 as *mut u8;
    unsafe {
        to.copy_from(from, 0); // this is fine
        libc::memcpy(to.cast(), from.cast(), 0); //~ERROR: dangling
    }
}
