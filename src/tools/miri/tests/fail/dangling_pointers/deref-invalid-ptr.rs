// This should fail even without validation.
//@compile-flags: -Zmiri-disable-validation -Zmiri-permissive-provenance

fn main() {
    let x = 16usize as *const u32;
    let _y = unsafe { &*x as *const u32 }; //~ ERROR: is a dangling pointer
}
