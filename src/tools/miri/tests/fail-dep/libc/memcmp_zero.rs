//@compile-flags: -Zmiri-permissive-provenance

// C says that passing "invalid" pointers is UB for all string functions.
// It is unclear whether `(int*)42` is "invalid", but there is no actually
// a `char` living at that address, so arguably it cannot be a valid pointer.
// Hence this is UB.
fn main() {
    let ptr = 42 as *const u8;
    unsafe {
        libc::memcmp(ptr.cast(), ptr.cast(), 0); //~ERROR: dangling
    }
}
