fn main() {
    let p = 42 as *const i32;
    let x = unsafe { *p }; //~ ERROR: attempted to interpret some raw bytes as a pointer address
    panic!("this should never print: {}", x);
}
