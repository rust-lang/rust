fn main() {
    let x: i32 = unsafe { *std::ptr::null() }; //~ ERROR: a memory access tried to interpret some bytes as a pointer
    panic!("this should never print: {}", x);
}
