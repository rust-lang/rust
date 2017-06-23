fn main() {
    let x = 0usize as *const u32;
    // This must fail because the pointer is NULL
    let _ = unsafe { &*x }; //~ ERROR: invalid use of NULL pointer
}
