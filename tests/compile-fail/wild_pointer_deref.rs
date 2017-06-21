fn main() {
    let p = 42 as *const i32;
    let x = unsafe { *p }; //~ ERROR: a memory access tried to interpret some bytes as a pointer
    panic!("this should never print: {}", x);
}
