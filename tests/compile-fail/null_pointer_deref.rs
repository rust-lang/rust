fn main() {
    let x: i32 = unsafe { *std::ptr::null() }; //~ ERROR: attempted to interpret some raw bytes as a pointer address
    panic!("this should never print: {}", x);
}
