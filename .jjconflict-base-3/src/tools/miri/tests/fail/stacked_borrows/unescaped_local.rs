//@compile-flags: -Zmiri-permissive-provenance

// Make sure we cannot use raw ptrs to access a local that
// we took the direct address of.
fn main() {
    let mut x = 42;
    let raw = &mut x as *mut i32 as usize as *mut i32;
    let _ptr = &mut x;
    unsafe {
        *raw = 13; //~ ERROR: /write access .* no exposed tags/
    }
}
