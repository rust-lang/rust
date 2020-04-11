// This should fail even without validation.
// compile-flags: -Zmiri-disable-validation

fn main() {
    let x = 16usize as *const u32;
    let _y = unsafe { &*x as *const u32 }; //~ ERROR invalid use of 16 as a pointer
}
