fn main() {
    let x: i32 = unsafe { *std::ptr::null() }; //~ ERROR: tried to access memory through an invalid pointer
    panic!("this should never print: {}", x);
}
