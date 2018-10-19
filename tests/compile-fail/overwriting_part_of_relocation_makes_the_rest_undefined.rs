fn main() {
    let mut p = &42;
    unsafe {
        let ptr: *mut _ = &mut p;
        *(ptr as *mut u8) = 123; // if we ever support 8 bit pointers, this is gonna cause
        // "attempted to interpret some raw bytes as a pointer address" instead of
        // "attempted to read undefined bytes"
    }
    let x = *p; //~ ERROR attempted to read undefined bytes
    panic!("this should never print: {}", x);
}
